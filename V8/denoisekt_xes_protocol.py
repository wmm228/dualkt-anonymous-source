"""Question-level adapters for the shared DenoiseKT-style input contract."""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def _row_normalize(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return sparse.diags(inverse) @ matrix


def _to_torch_sparse(matrix):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.stack([
        matrix.row.astype(np.int64),
        matrix.col.astype(np.int64),
    ]))
    values = torch.from_numpy(matrix.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, dtype=torch.float32
    ).coalesce()


def _build_question_structure(raw, materialize_question_graph):
    """Build shifted graph tensors from a binary question/KC matrix."""
    raw = raw.tocsr().astype(np.float32)
    raw.data[:] = 1.0
    raw.eliminate_zeros()
    num_questions, num_concepts = raw.shape

    raw_coo = raw.tocoo()
    shifted = sparse.coo_matrix(
        (
            raw_coo.data,
            (raw_coo.row + 1, raw_coo.col + 1),
        ),
        shape=(num_questions + 1, num_concepts + 1),
        dtype=np.float32,
    ).tocsr()
    incidence = _row_normalize(shifted)
    reverse_incidence = _row_normalize(shifted.transpose())

    question_graph = None
    question_graph_edges = int(sum(
        count * count
        for count in np.diff(shifted.tocsc().indptr).tolist()
    ))
    if materialize_question_graph:
        question_graph = shifted @ shifted.transpose()
        question_graph.data[:] = 1.0
        question_graph.eliminate_zeros()
        question_graph = _row_normalize(question_graph)
        question_graph_edges = int(question_graph.nnz)

    bundle_lookup = torch.zeros(num_questions + 1, dtype=torch.long)
    max_bundle_size = max(
        (stop - start for start, stop in zip(raw.indptr[:-1], raw.indptr[1:])),
        default=1,
    )
    concept_lookup = torch.zeros(
        num_questions + 1, max(max_bundle_size, 1), dtype=torch.long
    )
    bundle_ids = {}
    for raw_question in range(num_questions):
        start, stop = raw.indptr[raw_question:raw_question + 2]
        if start == stop:
            continue
        shifted_concepts = raw.indices[start:stop] + 1
        key = tuple(shifted_concepts.tolist())
        bundle_id = bundle_ids.setdefault(key, len(bundle_ids) + 1)
        bundle_lookup[raw_question + 1] = bundle_id
        concept_lookup[
            raw_question + 1, :len(shifted_concepts)
        ] = torch.from_numpy(shifted_concepts.astype(np.int64))

    return {
        'question_concept_incidence': _to_torch_sparse(incidence),
        'concept_question_incidence': _to_torch_sparse(reverse_incidence),
        'question_graph': (
            _to_torch_sparse(question_graph)
            if question_graph is not None else None
        ),
        'question_to_bundle': bundle_lookup,
        'question_to_concepts': concept_lookup,
        'num_bundles': len(bundle_ids),
        'question_graph_edges': question_graph_edges,
        'question_graph_materialized': question_graph is not None,
        'question_concept_edges': int(raw.nnz),
    }


def load_question_structure(qmatrix_path, num_questions, num_concepts):
    """Build label-free structure from a cached dense Q-matrix."""
    archive = np.load(Path(qmatrix_path))
    if 'matrix' not in archive.files:
        raise ValueError('qmatrix.npz must contain a matrix array')
    stored = archive['matrix']
    if stored.shape[0] < num_questions or stored.shape[1] < num_concepts:
        raise ValueError(
            f'qmatrix shape {stored.shape} is smaller than '
            f'({num_questions}, {num_concepts})'
        )
    raw = sparse.csr_matrix(
        stored[:num_questions, :num_concepts].astype(np.float32)
    )
    if np.any(np.diff(raw.indptr) == 0):
        raise ValueError('every real question must have at least one KC')
    return _build_question_structure(raw, materialize_question_graph=True)


def load_question_structure_from_metadata(
    metadata_paths,
    num_questions,
    num_concepts,
):
    """Build a response-free Q-matrix by unioning observed KC metadata per qid."""
    question_concepts = [set() for _ in range(num_questions)]
    for metadata_path in metadata_paths:
        path = Path(metadata_path)
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'questions', 'concepts'}.issubset(reader.fieldnames or []):
                raise ValueError(
                    f'{path} must contain questions and concepts columns'
                )
            for row_number, row in enumerate(reader, start=2):
                questions = row['questions'].split(',')
                concepts = row['concepts'].split(',')
                if len(questions) != len(concepts):
                    raise ValueError(
                        f'{path}:{row_number} has mismatched question/KC lengths'
                    )
                for question_token, concept_token in zip(questions, concepts):
                    question = int(question_token)
                    if question < 0:
                        continue
                    if question >= num_questions:
                        raise ValueError(
                            f'{path}:{row_number} question {question} is out of range'
                        )
                    for value in concept_token.split('_'):
                        concept = int(value)
                        if concept < 0:
                            continue
                        if concept >= num_concepts:
                            raise ValueError(
                                f'{path}:{row_number} KC {concept} is out of range'
                            )
                        question_concepts[question].add(concept)

    rows = []
    columns = []
    for question, concepts in enumerate(question_concepts):
        for concept in sorted(concepts):
            rows.append(question)
            columns.append(concept)
    raw = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(num_questions, num_concepts),
        dtype=np.float32,
    )
    if raw.nnz == 0:
        raise ValueError('question metadata contains no valid question/KC edges')
    # The bipartite form scales to datasets where a materialized q-q graph is huge.
    return _build_question_structure(raw, materialize_question_graph=False)


class IndexedDataset(Dataset):
    """Retain source row indices for compact per-question audit outputs."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample['sequence_index'] = torch.tensor(index, dtype=torch.long)
        return sample


class QuestionWindowIndexedDataset(IndexedDataset):
    """Attach pyKT question-window identities to every test sequence."""

    def __init__(self, dataset, metadata_path, folds):
        super().__init__(dataset)
        wanted_folds = {int(fold) for fold in folds}
        rows = {'qidxs': [], 'rests': [], 'orirows': []}
        with Path(metadata_path).open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'fold', 'qidxs', 'rest', 'orirow'}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f'{metadata_path} is missing question-window columns: '
                    f'{sorted(missing)}'
                )
            for row in reader:
                if int(float(row['fold'])) not in wanted_folds:
                    continue
                for source, target in [
                    ('qidxs', 'qidxs'),
                    ('rest', 'rests'),
                    ('orirow', 'orirows'),
                ]:
                    rows[target].append([
                        int(float(value))
                        for value in row[source].split(',')
                    ])
        if len(rows['qidxs']) != len(dataset):
            raise ValueError(
                f'question-window metadata rows ({len(rows["qidxs"])}) do not '
                f'match dataset rows ({len(dataset)})'
            )
        widths = {len(row) for row in rows['qidxs']}
        if len(widths) != 1:
            raise ValueError('question-window qidxs must have a uniform length')
        self.window_metadata = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in rows.items()
        }

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for key, values in self.window_metadata.items():
            sample[key] = values[index]
        return sample


class QuestionLevelCollator:
    """Convert pyKT tensors to V8 without dropping qid or multi-KC bundles."""

    def __init__(self, question_to_bundle, question_to_concepts=None):
        self.question_to_bundle = question_to_bundle.long()
        self.question_to_concepts = (
            question_to_concepts.long()
            if question_to_concepts is not None else None
        )

    def __call__(self, samples):
        data = default_collate(samples)
        raw_concepts = torch.cat(
            [data['cseqs'][:, :1], data['shft_cseqs']], dim=1
        )
        raw_questions = torch.cat(
            [data['qseqs'][:, :1], data['shft_qseqs']], dim=1
        )
        response_seq = torch.cat(
            [data['rseqs'][:, :1], data['shft_rseqs']], dim=1
        ).clamp(0.0, 1.0)

        valid_steps = data['masks'].bool()
        seq_len = valid_steps.long().sum(dim=1) + 1
        positions = torch.arange(raw_questions.size(1)).unsqueeze(0)
        event_valid = positions < seq_len.unsqueeze(1)
        concept_seq = torch.where(
            event_valid.unsqueeze(-1) & raw_concepts.ge(0),
            raw_concepts + 1,
            torch.zeros_like(raw_concepts),
        )
        question_seq = torch.where(
            event_valid & raw_questions.ge(0),
            raw_questions + 1,
            torch.zeros_like(raw_questions),
        )
        if self.question_to_concepts is not None:
            concept_seq = self.question_to_concepts[question_seq]
        transition_key_seq = self.question_to_bundle[question_seq]
        predict_mask = data['smasks'].bool() & valid_steps
        batch = {
            'concept_seq': concept_seq.long(),
            'question_seq': question_seq.long(),
            'transition_key_seq': transition_key_seq.long(),
            'response_seq': response_seq.float(),
            'seq_len': seq_len.long(),
            'predict_mask': predict_mask,
            'sequence_index': data['sequence_index'].long(),
        }
        for key in ('qidxs', 'rests', 'orirows'):
            if key in data:
                batch[key] = data[key][:, 1:].long()
        return batch


@torch.no_grad()
def evaluate_question_window_late_mean(model, loader, device, output_path):
    """Use pyKT's late_mean window fusion for a model with qid predictions."""
    from pykt.models.evaluate_model import group_fusion

    model.eval()
    history = {}
    fused = {'late_trues': [], 'late_mean': [], 'qidxs': [], 'row': []}
    # pyKT always writes a text row for every fused question. The compressed
    # NPZ below is the retained prediction artifact, so discard the duplicate
    # text stream without accumulating it in memory or on disk.
    with open(os.devnull, 'w', encoding='utf-8') as fusion_sink:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            probability = model.forward_sequence(batch)
            current = {
                'hs': [],
                'sm': batch['predict_mask'],
                'cq': batch['question_seq'],
                'cc': batch['question_seq'],
                'cr': batch['response_seq'],
                'y': probability,
                'qidxs': batch['qidxs'],
                'rests': batch['rests'],
                'orirow': batch['orirows'],
            }
            if history:
                merged = {'hs': []}
                for key in current:
                    if key != 'hs':
                        merged[key] = torch.cat([history[key], current[key]], dim=0)
            else:
                merged = current
            result, history = group_fusion(
                merged,
                model,
                'matra4kt',
                fusion_type=['late_fusion'],
                fout=fusion_sink,
            )
            for key in fused:
                if key in result:
                    fused[key].append(result[key])

    values = {
        key: np.concatenate(parts, axis=0)
        for key, parts in fused.items()
        if parts
    }
    if not {'late_trues', 'late_mean'}.issubset(values):
        raise RuntimeError('question-window late_mean evaluation produced no outputs')
    labels = values['late_trues'].astype(np.int64)
    probabilities = values['late_mean'].astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        raise RuntimeError('question-window late_mean evaluation lacks both labels')
    metrics = {
        'auc': float(roc_auc_score(labels, probabilities)),
        'acc': float(accuracy_score(labels, probabilities >= 0.5)),
        'count': int(labels.size),
        'aggregation': 'question_window_late_mean',
    }
    values['label'] = labels
    values['probability'] = probabilities.astype(np.float32)
    np.savez_compressed(Path(output_path), **values)
    return metrics


@torch.no_grad()
def export_question_predictions(model, loader, device, output_path, variant):
    """Save one auditable record for every selected next-question output."""
    model.eval()
    fields = {
        'sequence_index': [],
        'position': [],
        'question_id': [],
        'concept_ids': [],
        'label': [],
        'probability': [],
    }
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        probability = model.forward_sequence(batch)
        mask = batch['predict_mask']
        positions = torch.arange(
            probability.size(1), device=probability.device
        ).unsqueeze(0).expand_as(probability)
        sequence_indices = batch['sequence_index'].unsqueeze(1).expand_as(
            probability
        )
        raw_questions = batch['question_seq'][:, 1:] - 1
        raw_concepts = torch.where(
            batch['concept_seq'][:, 1:] > 0,
            batch['concept_seq'][:, 1:] - 1,
            torch.full_like(batch['concept_seq'][:, 1:], -1),
        )
        values = {
            'sequence_index': sequence_indices[mask].to(torch.int32),
            'position': (positions[mask] + 1).to(torch.int16),
            'question_id': raw_questions[mask].to(torch.int32),
            'concept_ids': raw_concepts[mask].to(torch.int16),
            'label': batch['response_seq'][:, 1:][mask].to(torch.uint8),
            'probability': probability[mask].to(torch.float32),
        }
        for key, value in values.items():
            fields[key].append(value.cpu().numpy())

    payload = {
        key: np.concatenate(values, axis=0)
        for key, values in fields.items()
    }
    payload['input_variant'] = np.asarray(str(variant))
    np.savez_compressed(Path(output_path), **payload)
    return int(payload['label'].size)
