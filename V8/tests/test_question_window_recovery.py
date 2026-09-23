import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from V8.denoisekt_xes_protocol import evaluate_question_window_late_mean
from V8.train_denoisekt_xes import load_recoverable_training


class _Model:
    def eval(self):
        return self

    def forward_sequence(self, batch):
        return torch.full_like(batch['response_seq'], 0.5, dtype=torch.float32)


def test_question_window_fusion_receives_writable_sink(tmp_path):
    seen = []

    def fake_group_fusion(data, model, model_name, fusion_type, fout):
        del data, model, model_name, fusion_type
        assert fout is not None
        seen.append(fout.write('discarded duplicate row\n'))
        return {
            'late_trues': np.asarray([0, 1]),
            'late_mean': np.asarray([0.2, 0.8]),
            'qidxs': np.asarray([10, 11]),
            'row': np.asarray([0, 1]),
        }, {}

    batch = {
        key: torch.ones((1, 2), dtype=torch.long)
        for key in (
            'question_seq', 'response_seq', 'qidxs', 'rests', 'orirows'
        )
    }
    batch['predict_mask'] = torch.ones((1, 2), dtype=torch.bool)
    output = tmp_path / 'predictions.npz'
    with patch('pykt.models.evaluate_model.group_fusion', fake_group_fusion):
        metrics = evaluate_question_window_late_mean(
            _Model(), [batch], 'cpu', output
        )

    assert seen and seen[0] > 0
    assert metrics == {
        'auc': 1.0,
        'acc': 1.0,
        'count': 2,
        'aggregation': 'question_window_late_mean',
    }
    with np.load(output) as values:
        assert values['label'].tolist() == [0, 1]
        assert np.allclose(values['probability'], [0.2, 0.8])


def test_recovery_requires_finished_training_and_matching_checkpoint(tmp_path):
    config = {
        'experiment': {
            'protocol': 'fixed_matra_real_history_length_5fold',
            'input_variant': 'single_denoised_orthogonal_fusion',
        },
        'training': {'epochs': 200, 'patience': 10},
    }
    checkpoint = tmp_path / 'fold3_best.pt'
    progress = tmp_path / 'fold3_progress.json'
    torch.save({
        'model_state_dict': {'weight': torch.ones(1)},
        'best_epoch': 5,
        'fold': 3,
        **config['experiment'],
    }, checkpoint)
    payload = {
        'completed_epochs': 15,
        'early_stop_counter': 10,
        'history': {
            'best_epoch': 5,
            'test_auc': [0.1],
            'test_acc': [0.2],
            'epoch_duration_seconds': [1.0, 2.0],
        },
    }
    progress.write_text(json.dumps(payload), encoding='utf-8')

    loaded, _, history = load_recoverable_training(
        checkpoint, progress, config, 3
    )
    assert loaded['best_epoch'] == 5
    assert history['test_auc'] == []
    assert history['test_acc'] == []

    payload['early_stop_counter'] = 9
    progress.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(RuntimeError, match='training did not finish'):
        load_recoverable_training(checkpoint, progress, config, 3)
