"""Modular, inductive knowledge tracing from universal KT fields.

Every optional module consumes only concept IDs, prefix responses, or
training-fold statistics derived from those fields. Modules return a state and
an evidence reliability; they never own prediction heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from V7.model import (
    ResidualMambaEncoder,
    TargetCrossAttentionLayer,
    TargetCrossAttentionTransformer,
)


class DifficultyAwareConceptEncoder(nn.Module):
    """Add a confidence-weighted train-fold difficulty attribute to concepts."""

    def __init__(
        self,
        n_concepts,
        d_model,
        concept_statistics=None,
        dropout=0.2,
        enabled=True,
    ):
        super().__init__()
        if concept_statistics is None:
            concept_statistics = torch.zeros(n_concepts, 2)
            concept_statistics[:, 0] = 0.5
        if tuple(concept_statistics.shape) != (n_concepts, 2):
            raise ValueError(
                'concept_statistics must have [difficulty, confidence] '
                'for every concept'
            )
        self.enabled = bool(enabled)
        self.embedding = nn.Embedding(n_concepts, d_model, padding_idx=0)
        self.register_buffer(
            'difficulty', concept_statistics[:, 0].float().clamp(0.0, 1.0)
        )
        self.register_buffer(
            'confidence', concept_statistics[:, 1].float().clamp(0.0, 1.0)
        )
        self.attribute_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self):
        base = self.embedding.weight
        if self.enabled:
            attributes = torch.stack(
                [self.difficulty, self.confidence], dim=-1
            )
            attribute_state = self.attribute_encoder(attributes)
            base = base + self.dropout(attribute_state) * self.confidence.unsqueeze(-1)
        output = self.output_norm(base)
        padding_mask = output.new_ones(output.size(0), 1)
        padding_mask[0] = 0.0
        return output * padding_mask

    def success_probability(self):
        if self.enabled:
            return 1.0 - self.difficulty
        return self.difficulty.new_full(self.difficulty.shape, 0.5)


class StaticQuestionGraphEncoder(nn.Module):
    """Encode question identity with label-free question/KC metadata."""

    def __init__(
        self,
        n_questions,
        n_concepts,
        d_model,
        question_graph,
        question_concept_incidence,
        concept_question_incidence=None,
        use_question_rasch=False,
        dropout=0.2,
    ):
        super().__init__()
        if n_questions < 2:
            raise ValueError('question context requires a question vocabulary')
        if question_graph is not None and tuple(question_graph.shape) != (
            n_questions, n_questions
        ):
            raise ValueError(
                'question_graph must have shape [questions, questions]'
            )
        if question_concept_incidence is None or tuple(
            question_concept_incidence.shape
        ) != (n_questions, n_concepts):
            raise ValueError(
                'question_concept_incidence must have shape '
                '[questions, concepts]'
            )
        if question_graph is not None and not question_graph.is_sparse:
            raise ValueError('question_graph must be a sparse tensor')
        if not question_concept_incidence.is_sparse:
            raise ValueError(
                'question_concept_incidence must be a sparse tensor'
            )
        if concept_question_incidence is not None:
            if tuple(concept_question_incidence.shape) != (
                n_concepts, n_questions
            ):
                raise ValueError(
                    'concept_question_incidence must have shape '
                    '[concepts, questions]'
                )
            if not concept_question_incidence.is_sparse:
                raise ValueError(
                    'concept_question_incidence must be a sparse tensor'
                )
        if question_graph is None and concept_question_incidence is None:
            raise ValueError(
                'question encoding requires a q-q graph or reverse incidence'
            )
        self.embedding = nn.Embedding(
            n_questions, d_model, padding_idx=0
        )
        self.register_buffer(
            'question_graph',
            question_graph.coalesce() if question_graph is not None else None,
        )
        self.register_buffer(
            'question_concept_incidence',
            question_concept_incidence.coalesce(),
        )
        self.register_buffer(
            'concept_question_incidence',
            (
                concept_question_incidence.coalesce()
                if concept_question_incidence is not None else None
            ),
        )
        self.use_question_rasch = bool(use_question_rasch)
        # Always allocate this scalar so ablations retain identical RNG streams.
        self.question_difficulty = nn.Embedding(
            n_questions, 1, padding_idx=0
        )
        nn.init.zeros_(self.question_difficulty.weight)
        self.graph_projection = nn.Linear(d_model, d_model, bias=False)
        self.concept_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, concept_states):
        base = self.embedding.weight
        if self.question_graph is not None:
            graph_state = torch.sparse.mm(self.question_graph, base)
        else:
            graph_state = torch.sparse.mm(
                self.question_concept_incidence,
                torch.sparse.mm(self.concept_question_incidence, base),
            )
        concept_state = torch.sparse.mm(
            self.question_concept_incidence, concept_states
        )
        rasch_state = 0.0
        if self.use_question_rasch:
            rasch_state = torch.tanh(
                self.question_difficulty.weight
            ) * concept_state
        output = self.output_norm(
            base
            + self.dropout(self.graph_projection(graph_state))
            + self.dropout(self.concept_projection(concept_state))
            + rasch_state
        )
        padding_mask = output.new_ones(output.size(0), 1)
        padding_mask[0] = 0.0
        return output * padding_mask


class HierarchicalItemEvidence(nn.Module):
    """Training-fold PID residual relative to an item's concept bundle."""

    def __init__(
        self,
        n_items,
        n_concepts,
        d_model,
        item_statistics,
        item_concept_incidence,
        dropout=0.2,
    ):
        super().__init__()
        if item_statistics is None or tuple(item_statistics.shape) != (
            n_items, 3
        ):
            raise ValueError(
                'item_statistics must contain difficulty, confidence, and '
                'frequency for every item'
            )
        if item_concept_incidence is None or tuple(
            item_concept_incidence.shape
        ) != (n_items, n_concepts):
            raise ValueError(
                'item_concept_incidence must have shape [items, concepts]'
            )
        self.embedding = nn.Embedding(n_items, d_model, padding_idx=0)
        self.register_buffer(
            'difficulty', item_statistics[:, 0].float().clamp(0.0, 1.0)
        )
        self.register_buffer(
            'confidence', item_statistics[:, 1].float().clamp(0.0, 1.0)
        )
        self.available = bool(torch.any(self.confidence > 0).item())
        self.register_buffer(
            'frequency', item_statistics[:, 2].float().clamp(0.0, 1.0)
        )
        self.register_buffer(
            'item_concept_incidence', item_concept_incidence.float()
        )
        self.bundle_projection = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attribute_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, item_ids, target_ids, concept_states, concept_success):
        weights = self.item_concept_incidence[item_ids]
        bundle_state = torch.matmul(weights, concept_states)
        target_state = concept_states[target_ids]
        bundle_success = torch.matmul(
            weights, concept_success.unsqueeze(-1)
        ).squeeze(-1)
        item_success = 1.0 - self.difficulty[item_ids]
        attributes = torch.stack([
            item_success,
            bundle_success,
            item_success - bundle_success,
            self.frequency[item_ids],
        ], dim=-1)
        bundle = self.bundle_projection(torch.cat([
            bundle_state, target_state, bundle_state * target_state
        ], dim=-1))
        state = self.output_norm(
            self.embedding(item_ids) + bundle + self.attribute_encoder(attributes)
        )
        reliability = self.confidence[item_ids].unsqueeze(-1)
        return state, reliability


class CausalRaschAbility(nn.Module):
    """Infer one online learner ability from difficulty-calibrated responses."""

    def __init__(
        self,
        d_model,
        prior_precision=1.0,
        reliability_strength=5.0,
        refinement_steps=2,
        dropout=0.2,
    ):
        super().__init__()
        self.prior_precision = float(prior_precision)
        self.reliability_strength = float(reliability_strength)
        self.refinement_steps = int(refinement_steps)
        if self.prior_precision <= 0.0:
            raise ValueError('Rasch prior_precision must be positive')
        if self.refinement_steps < 1:
            raise ValueError('Rasch refinement_steps must be positive')
        self.innovation_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.evidence_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _concept_difficulty(success):
        success = success.float().clamp(1e-4, 1.0 - 1e-4)
        return -torch.logit(success)

    def _initial_ability(self, batch, difficulty):
        batch_size = batch['concept_seq'].size(0)
        n_concepts = difficulty.numel()
        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None:
            attempts = difficulty.new_zeros(batch_size, n_concepts)
        if correct is None:
            correct = difficulty.new_zeros(batch_size, n_concepts)
        attempts = attempts.to(difficulty.dtype)
        correct = correct.to(difficulty.dtype)

        theta = difficulty.new_zeros(batch_size)
        for _ in range(6):
            probability = torch.sigmoid(
                theta.unsqueeze(-1) - difficulty.unsqueeze(0)
            )
            score = (
                correct - attempts * probability
            ).sum(dim=-1) - self.prior_precision * theta
            information = self.prior_precision + (
                attempts * probability * (1.0 - probability)
            ).sum(dim=-1)
            theta = (theta + score / information.clamp_min(1e-4)).clamp(
                -6.0, 6.0
            )

        probability = torch.sigmoid(
            theta.unsqueeze(-1) - difficulty.unsqueeze(0)
        )
        information = self.prior_precision + (
            attempts * probability * (1.0 - probability)
        ).sum(dim=-1)
        return theta, information, attempts.sum(dim=-1)

    def _trace(self, batch, success):
        concepts = batch['concept_seq']
        responses = batch['response_seq'].to(success.dtype)
        positions = torch.arange(concepts.size(1), device=concepts.device)
        valid = positions.unsqueeze(0) < batch['seq_len'].unsqueeze(1)
        valid_float = valid.to(success.dtype)
        difficulty = self._concept_difficulty(success).to(responses.dtype)
        event_difficulty = difficulty[concepts]
        initial_theta, initial_information, initial_count = (
            self._initial_ability(batch, difficulty)
        )

        theta_after = initial_theta.unsqueeze(1).expand_as(responses)
        expected = torch.sigmoid(
            initial_theta.unsqueeze(1) - event_difficulty
        )
        for _ in range(self.refinement_steps):
            theta_before = torch.cat([
                initial_theta.unsqueeze(1), theta_after[:, :-1]
            ], dim=1)
            expected = torch.sigmoid(theta_before - event_difficulty)
            innovation = (responses - expected) * valid_float
            information = expected * (1.0 - expected) * valid_float
            theta_after = (
                initial_theta.unsqueeze(1)
                + innovation.cumsum(dim=1)
                / (
                    initial_information.unsqueeze(1)
                    + information.cumsum(dim=1)
                ).clamp_min(1e-4)
            ).clamp(-6.0, 6.0)

        theta_before = torch.cat([
            initial_theta.unsqueeze(1), theta_after[:, :-1]
        ], dim=1)
        expected = torch.sigmoid(theta_before - event_difficulty)
        innovation = (responses - expected) * valid_float
        fisher = expected * (1.0 - expected) * valid_float
        count_after = initial_count.unsqueeze(1) + valid_float.cumsum(dim=1)
        reliability = count_after / (
            count_after + self.reliability_strength
        )
        reliability = reliability * valid_float
        return {
            'difficulty': difficulty,
            'theta_before': theta_before,
            'theta_after': theta_after,
            'expected': expected,
            'innovation': innovation,
            'fisher': fisher,
            'reliability': reliability,
            'valid': valid,
        }

    def _innovation_state(self, trace):
        features = torch.stack([
            trace['innovation'],
            trace['expected'] - 0.5,
            4.0 * trace['fisher'],
            torch.tanh(trace['theta_before'] / 3.0),
            trace['reliability'],
        ], dim=-1)
        state = self.innovation_encoder(features)
        state = state.masked_fill(~trace['valid'].unsqueeze(-1), 0.0)
        return state, trace['reliability'].unsqueeze(-1)

    def _target_state(self, theta, reliability, target_ids, difficulty):
        target_difficulty = difficulty[target_ids]
        expected = torch.sigmoid(theta - target_difficulty)
        features = torch.stack([
            torch.tanh(theta / 3.0),
            torch.tanh(target_difficulty / 3.0),
            expected - 0.5,
            reliability,
        ], dim=-1)
        state = self.evidence_encoder(features)
        reliability = reliability.unsqueeze(-1)
        return state * reliability, reliability

    def forward_sequence(self, batch, success):
        trace = self._trace(batch, success)
        target_ids = batch['concept_seq'][:, 1:]
        evidence = self._target_state(
            trace['theta_after'][:, :-1],
            trace['reliability'][:, :-1],
            target_ids,
            trace['difficulty'],
        )
        return {
            'innovation': self._innovation_state(trace),
            'evidence': evidence,
            'theta_after': trace['theta_after'],
        }

    def forward_target(self, batch, success):
        trace = self._trace(batch, success)
        index = (batch['seq_len'] - 1).clamp_min(0)
        rows = torch.arange(index.size(0), device=index.device)
        theta = trace['theta_after'][rows, index]
        reliability = trace['reliability'][rows, index]
        evidence = self._target_state(
            theta,
            reliability,
            batch['target_concept'],
            trace['difficulty'],
        )
        return {
            'innovation': self._innovation_state(trace),
            'evidence': evidence,
            'theta_after': trace['theta_after'],
        }


class RaschInnovationAdapter(nn.Module):
    """Inject learner-relative response surprise before temporal encoding."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, events, innovation, reliability):
        gate = self.gate(torch.cat([events, innovation], dim=-1))
        effective = gate * reliability.clamp(0.0, 1.0)
        return self.output_norm(
            events + self.dropout(effective * self.projection(innovation))
        )


class PopulationConceptHypergraph(nn.Module):
    """Propagate a train-fold population hypergraph between concept nodes."""

    def __init__(self, graph, d_model, dropout=0.2, enabled=True):
        super().__init__()
        if graph.ndim != 2 or graph.size(0) != graph.size(1):
            raise ValueError('concept hypergraph operator must be square')
        self.enabled = bool(enabled)
        self.register_buffer('graph', graph.float())
        self.input_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, concept_states):
        if not self.enabled:
            return concept_states
        transformed = self.projection(self.input_norm(concept_states))
        propagated = (
            torch.sparse.mm(self.graph, transformed)
            if self.graph.is_sparse else self.graph @ transformed
        )
        output = self.output_norm(
            concept_states + self.dropout(F.gelu(propagated))
        )
        padding_mask = output.new_ones(output.size(0), 1)
        padding_mask[0] = 0.0
        return output * padding_mask


class PrefixPopulationEvidence(nn.Module):
    """Read train-only concept graph states through an observed learner prefix."""

    def __init__(self, d_model, reliability_strength=5.0, dropout=0.2):
        super().__init__()
        self.reliability_strength = float(reliability_strength)
        self.encoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _initial_memory(batch, concept_states, success):
        batch_size = batch['concept_seq'].size(0)
        n_concepts = concept_states.size(0)
        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None:
            attempts = concept_states.new_zeros(batch_size, n_concepts)
        if correct is None:
            correct = concept_states.new_zeros(batch_size, n_concepts)
        attempts = attempts.to(concept_states.dtype)
        correct = correct.to(concept_states.dtype)
        exposure = attempts @ concept_states
        calibrated = correct - attempts * success.unsqueeze(0)
        outcome = calibrated @ concept_states
        return exposure, outcome, attempts.sum(dim=-1, keepdim=True)

    def forward_sequence(self, batch, concept_states, success):
        concepts = batch['concept_seq']
        responses = batch['response_seq'].to(concept_states.dtype)
        positions = torch.arange(concepts.size(1), device=concepts.device)
        valid = positions.unsqueeze(0) < batch['seq_len'].unsqueeze(1)
        event_concepts = concept_states[concepts] * valid.unsqueeze(-1)

        exposure, outcome, initial_count = self._initial_memory(
            batch, concept_states, success
        )
        exposure = exposure.unsqueeze(1) + event_concepts.cumsum(dim=1)
        residual = responses - success[concepts]
        residual = residual * valid.to(residual.dtype)
        outcome = outcome.unsqueeze(1) + (
            residual.unsqueeze(-1) * event_concepts
        ).cumsum(dim=1)
        count = initial_count.unsqueeze(1) + valid.cumsum(dim=1).unsqueeze(-1)
        state = self.encoder(torch.cat([
            exposure / count.clamp_min(1.0),
            outcome / count.clamp_min(1.0),
        ], dim=-1))
        reliability = count / (count + self.reliability_strength)
        state = state.masked_fill(~valid.unsqueeze(-1), 0.0)
        reliability = reliability.masked_fill(~valid.unsqueeze(-1), 0.0)
        return state, reliability


class TargetMasteryEvidence(nn.Module):
    """Difficulty-calibrated target mastery with zero cold-start reliability."""

    def __init__(self, d_model, prior_strength=5.0, dropout=0.2):
        super().__init__()
        self.prior_strength = float(prior_strength)
        self.encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, target_ids, attempts, correct, success):
        attempts = attempts.to(success.dtype)
        correct = correct.to(success.dtype)
        prior = success[target_ids]
        posterior = (
            correct + self.prior_strength * prior
        ) / (attempts + self.prior_strength)
        reliability = attempts / (attempts + self.prior_strength)
        delta = posterior - prior
        state = self.encoder(torch.stack([delta, reliability], dim=-1))
        reliability = reliability.unsqueeze(-1)
        return state * reliability, reliability


class DynamicConceptMemory(nn.Module):
    """Build learned target-indexed states with a parallel segmented scan."""

    def __init__(self, d_model, prior_strength=5.0, dropout=0.2):
        super().__init__()
        self.prior_strength = float(prior_strength)
        self.initial_encoder = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(3 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def _initial_state(self, batch, concept_states, success):
        batch_size = batch['concept_seq'].size(0)
        n_concepts = concept_states.size(0)
        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None:
            attempts = concept_states.new_zeros(batch_size, n_concepts)
        if correct is None:
            correct = concept_states.new_zeros(batch_size, n_concepts)
        attempts = attempts.to(concept_states.dtype)
        correct = correct.to(concept_states.dtype)
        prior = success.view(1, -1)
        posterior = (
            correct + self.prior_strength * prior
        ) / (attempts + self.prior_strength)
        reliability = attempts / (attempts + self.prior_strength)
        concept = concept_states.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([
            concept,
            (posterior - prior).unsqueeze(-1),
            reliability.unsqueeze(-1),
        ], dim=-1)
        state = self.initial_encoder(features)
        return state, attempts

    def _event_states(
        self, batch, concept_states, initial_state, initial_attempts, events
    ):
        concepts = batch['concept_seq']
        positions = torch.arange(concepts.size(1), device=concepts.device)
        valid = positions.unsqueeze(0) < batch['seq_len'].unsqueeze(1)
        updates = self.event_encoder(events)
        batch_offset = (
            torch.arange(concepts.size(0), device=concepts.device)
            * concept_states.size(0)
        ).unsqueeze(1)
        prefix_mean, prefix_count = (
            CausalTargetTransitionGraph._segmented_prefix_stats(
                updates, batch_offset + concepts, valid
            )
        )
        batch_index = torch.arange(concepts.size(0), device=concepts.device)
        batch_index = batch_index.unsqueeze(1).expand_as(concepts)
        prior_state = initial_state[batch_index, concepts]
        prior_count = initial_attempts[batch_index, concepts].unsqueeze(-1)
        total_count = prefix_count + prior_count
        total_mean = (
            prefix_mean * prefix_count + prior_state * prior_count
        ) / total_count.clamp_min(1.0)
        reliability = total_count / (total_count + self.prior_strength)
        state = self.output_norm(self.state_encoder(torch.cat([
            total_mean,
            updates,
            concept_states[concepts],
            reliability,
        ], dim=-1)))
        return state.masked_fill(~valid.unsqueeze(-1), 0.0)

    @staticmethod
    def _prefix_indices_and_counts(
        concepts, seq_len, initial_attempts, targets, include_all_events=False
    ):
        batch_size, length = concepts.shape
        n_concepts = initial_attempts.size(1)
        last = torch.full(
            (batch_size, n_concepts),
            -1,
            dtype=torch.long,
            device=concepts.device,
        )
        counts = initial_attempts.clone()
        batch = torch.arange(batch_size, device=concepts.device)
        output_last = []
        output_count = []
        steps = length if include_all_events else max(length - 1, 0)
        for position in range(steps):
            source = concepts[:, position]
            valid = position < seq_len
            old_last = last[batch, source]
            new_last = torch.where(
                valid, old_last.new_full(old_last.shape, position), old_last
            )
            last.scatter_(1, source.unsqueeze(1), new_last.unsqueeze(1))
            counts.scatter_add_(
                1,
                source.unsqueeze(1),
                valid.to(counts.dtype).unsqueeze(1),
            )
            if not include_all_events:
                target = targets[:, position]
                output_last.append(last[batch, target])
                output_count.append(counts[batch, target])
        if include_all_events:
            return last[batch, targets], counts[batch, targets]
        if not output_last:
            empty_index = concepts.new_zeros(batch_size, 0)
            empty_count = initial_attempts.new_zeros(batch_size, 0)
            return empty_index, empty_count
        return torch.stack(output_last, dim=1), torch.stack(output_count, dim=1)

    def _read_sequence(
        self, event_states, initial_state, target_ids, last_indices, counts
    ):
        batch = torch.arange(event_states.size(0), device=event_states.device)
        batch = batch.unsqueeze(1).expand_as(last_indices)
        gathered = event_states[batch, last_indices.clamp_min(0)]
        initial = initial_state[batch, target_ids]
        state = torch.where(
            (last_indices >= 0).unsqueeze(-1), gathered, initial
        )
        reliability = counts / (counts + self.prior_strength)
        state = state * (counts > 0).to(state.dtype).unsqueeze(-1)
        return state, reliability.unsqueeze(-1)

    def forward_sequence(self, batch, concept_states, success, events):
        concepts = batch['concept_seq']
        steps = max(concepts.size(1) - 1, 0)
        initial_state, initial_attempts = self._initial_state(
            batch, concept_states, success
        )
        if steps == 0:
            empty = concept_states.new_zeros(
                concepts.size(0), 0, concept_states.size(-1)
            )
            return empty, empty[..., :1]
        event_states = self._event_states(
            batch,
            concept_states,
            initial_state,
            initial_attempts,
            events,
        )
        target_ids = concepts[:, 1:]
        last, counts = self._prefix_indices_and_counts(
            concepts,
            batch['seq_len'],
            initial_attempts,
            target_ids,
        )
        state, reliability = self._read_sequence(
            event_states, initial_state, target_ids, last, counts
        )
        positions = torch.arange(steps, device=concepts.device)
        valid = positions.unsqueeze(0) < (batch['seq_len'] - 1).unsqueeze(1)
        return (
            state.masked_fill(~valid.unsqueeze(-1), 0.0),
            reliability.masked_fill(~valid.unsqueeze(-1), 0.0),
        )

    def forward_target(self, batch, concept_states, success, events):
        concepts = batch['concept_seq']
        initial_state, initial_attempts = self._initial_state(
            batch, concept_states, success
        )
        event_states = self._event_states(
            batch,
            concept_states,
            initial_state,
            initial_attempts,
            events,
        )
        targets = batch['target_concept']
        last, counts = self._prefix_indices_and_counts(
            concepts,
            batch['seq_len'],
            initial_attempts,
            targets,
            include_all_events=True,
        )
        batch_index = torch.arange(concepts.size(0), device=concepts.device)
        gathered = event_states[batch_index, last.clamp_min(0)]
        initial = initial_state[batch_index, targets]
        state = torch.where((last >= 0).unsqueeze(-1), gathered, initial)
        reliability = counts / (counts + self.prior_strength)
        state = state * (counts > 0).to(state.dtype).unsqueeze(-1)
        return state, reliability.unsqueeze(-1)


class CoverageAdaptiveFusion(nn.Module):
    """Let reliable target memory replace, rather than only perturb, global state."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, global_state, target, local_state, reliability):
        local_candidate = self.candidate(torch.cat([
            local_state, target, local_state * target
        ], dim=-1))
        learned = self.gate(torch.cat([
            global_state, target, local_state
        ], dim=-1))
        effective = learned * reliability.clamp(0.0, 1.0)
        fused = global_state + effective * (local_candidate - global_state)
        return self.output_norm(fused), effective


class PopulationBackboneAdapter(nn.Module):
    """Condition temporal concept tokens on train-only population structure."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.delta_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.last_gate = None

    def forward(self, base, population):
        delta = self.delta_projection(population - base)
        gate = self.gate(torch.cat([base, population, base * population], dim=-1))
        output = self.output_norm(base + self.dropout(gate * delta))
        padding_mask = output.new_ones(output.size(0), 1)
        padding_mask[0] = 0.0
        self.last_gate = gate.detach()
        return output * padding_mask


class TargetConditionedCausalRetrieval(nn.Module):
    """Retrieve target-relevant events without compressing them into one state."""

    def __init__(
        self,
        d_model,
        n_heads=4,
        reliability_strength=5.0,
        dropout=0.2,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by retrieval heads')
        self.n_heads = int(n_heads)
        self.reliability_strength = float(reliability_strength)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            self.n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # softplus(-2.25) is about 0.10, a weak initial recency preference.
        self.recency_logit = nn.Parameter(torch.full((self.n_heads,), -2.25))
        self.output_encoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.last_recency_decay = None

    def _attention_mask(
        self,
        batch_size,
        query_length,
        memory_length,
        seq_len,
        target_ids=None,
        memory_ids=None,
        scope='all',
    ):
        device = seq_len.device
        query_positions = torch.arange(query_length, device=device)
        memory_positions = torch.arange(memory_length, device=device)
        distance = query_positions[:, None] - memory_positions[None, :]
        causal = distance >= 0
        distance = distance.clamp_min(0).to(self.recency_logit.dtype)
        decay = F.softplus(self.recency_logit)
        bias = -decay[:, None, None] * torch.log1p(distance)[None, :, :]

        valid_length = seq_len.clamp_min(1).clamp_max(memory_length)
        memory_valid = (
            memory_positions.unsqueeze(0) < valid_length.unsqueeze(1)
        )
        allowed = causal.unsqueeze(0) & memory_valid.unsqueeze(1)
        if scope == 'same':
            if target_ids is None or memory_ids is None:
                raise ValueError('same-scope retrieval requires concept IDs')
            allowed = allowed & (
                target_ids.unsqueeze(-1) == memory_ids.unsqueeze(1)
            )
        count = allowed.sum(dim=-1, keepdim=True)
        safe_allowed = allowed.clone()
        empty = count.squeeze(-1) == 0
        safe_allowed[..., 0] = safe_allowed[..., 0] | empty
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        bias = bias.masked_fill(
            ~safe_allowed[:, None, :, :], float('-inf')
        )
        self.last_recency_decay = decay.detach()
        return (
            bias.reshape(
                batch_size * self.n_heads, query_length, memory_length
            ),
            count,
        )

    def _read(self, query, memory, attention_mask):
        context, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.output_encoder(torch.cat([
            context, query, context * query
        ], dim=-1))

    def forward_sequence(
        self,
        target,
        memory,
        seq_len,
        target_ids=None,
        memory_ids=None,
        scope='all',
    ):
        steps = target.size(1)
        if steps == 0:
            return target, target[..., :1]
        mask, count = self._attention_mask(
            target.size(0),
            steps,
            memory.size(1),
            seq_len,
            target_ids=target_ids,
            memory_ids=memory_ids,
            scope=scope,
        )
        state = self._read(target, memory, mask)
        positions = torch.arange(steps, device=target.device)
        valid = positions.unsqueeze(0) < (seq_len - 1).clamp_min(0).unsqueeze(1)
        count = count.to(state.dtype)
        reliability = count / (count + self.reliability_strength)
        state = state * (count > 0).to(state.dtype)
        state = state.masked_fill(~valid.unsqueeze(-1), 0.0)
        reliability = reliability.masked_fill(~valid.unsqueeze(-1), 0.0)
        return state, reliability

    def forward_target(
        self,
        target,
        memory,
        seq_len,
        target_ids=None,
        memory_ids=None,
        scope='all',
    ):
        query = target.unsqueeze(1)
        memory_positions = torch.arange(memory.size(1), device=memory.device)
        valid_length = seq_len.clamp_min(1).clamp_max(memory.size(1))
        memory_valid = memory_positions.unsqueeze(0) < valid_length.unsqueeze(1)
        allowed = memory_valid
        if scope == 'same':
            if target_ids is None or memory_ids is None:
                raise ValueError('same-scope retrieval requires concept IDs')
            allowed = allowed & (memory_ids == target_ids.unsqueeze(1))
        count = allowed.sum(dim=-1, keepdim=True)
        safe_allowed = allowed.clone()
        empty = count.squeeze(-1) == 0
        safe_allowed[:, 0] = safe_allowed[:, 0] | empty
        distance = seq_len.clamp_min(1).unsqueeze(1) - 1 - memory_positions
        distance = distance.clamp_min(0).to(self.recency_logit.dtype)
        decay = F.softplus(self.recency_logit)
        bias = -decay[None, :, None, None] * torch.log1p(
            distance[:, None, None, :]
        )
        bias = bias.masked_fill(
            ~safe_allowed[:, None, None, :], float('-inf')
        )
        self.last_recency_decay = decay.detach()
        state = self._read(
            query,
            memory,
            bias.reshape(target.size(0) * self.n_heads, 1, memory.size(1)),
        ).squeeze(1)
        count = count.to(state.dtype)
        reliability = count / (count + self.reliability_strength)
        return state * (count > 0).to(state.dtype), reliability


class CoverageAdaptiveHistoryRouter(nn.Module):
    """Route global, broad, and exact-target history by observed coverage."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.broad_projection = nn.Linear(d_model, d_model, bias=False)
        self.local_projection = nn.Linear(d_model, d_model, bias=False)
        self.router = nn.Sequential(
            nn.Linear(5 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
        self.output_norm = nn.LayerNorm(d_model)
        self.last_weights = None

    def forward(self, global_state, target, broad, local, local_reliability):
        broad = self.broad_projection(broad)
        local = self.local_projection(local)
        coverage = local_reliability.clamp(0.0, 1.0)
        logits = self.router(torch.cat([
            global_state,
            broad,
            local,
            target,
            global_state * target,
            coverage,
        ], dim=-1))
        availability = torch.cat([
            torch.ones_like(coverage),
            (1.0 - coverage).clamp_min(1e-4),
            coverage,
        ], dim=-1)
        weights = torch.softmax(
            logits + torch.log(availability.clamp_min(1e-8)), dim=-1
        )
        active = (availability > 0).to(weights.dtype)
        weights = weights * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = (
            weights[..., 0:1] * global_state
            + weights[..., 1:2] * broad
            + weights[..., 2:3] * local
        )
        self.last_weights = weights.detach()
        return self.output_norm(fused), weights


class TargetConditionedFactorizedFusion(nn.Module):
    """Fuse exposure and acquisition dynamics for the current target."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.exposure_adapter = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.acquisition_adapter = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.last_gate = None

    def forward(self, exposure, acquisition, target):
        exposure = self.exposure_adapter(torch.cat([
            exposure, target, exposure * target
        ], dim=-1))
        acquisition = self.acquisition_adapter(torch.cat([
            acquisition, target, acquisition * target
        ], dim=-1))
        gate = self.gate(torch.cat([
            exposure, acquisition, target, exposure * acquisition
        ], dim=-1))
        interaction = torch.tanh(
            self.cross_projection(exposure * acquisition)
        )
        state = (
            (1.0 - gate) * exposure
            + gate * acquisition
            + self.dropout(interaction) / math.sqrt(2.0)
        )
        self.last_gate = gate.detach()
        return self.output_norm(state)


class CausalTargetTransitionGraph(nn.Module):
    """Aggregate difficulty-calibrated prefix edges entering each target."""

    def __init__(
        self,
        d_model,
        reliability_strength=5.0,
        dropout=0.2,
        aggregation='uniform',
        decay=0.97,
    ):
        super().__init__()
        if aggregation not in {
            'uniform', 'recency', 'occurrence_recency'
        }:
            raise ValueError(
                'transition aggregation must be uniform, recency, or '
                'occurrence_recency'
            )
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError('transition decay must be in (0, 1]')
        self.reliability_strength = float(reliability_strength)
        self.aggregation = aggregation
        self.decay = float(decay)
        self.edge_projection = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.last_edge_gate = None

    def _edge_states(self, source, destination, outcome):
        features = torch.cat([
            source,
            destination,
            outcome,
            source * destination,
        ], dim=-1)
        hidden = self.edge_projection(features)
        gate = self.edge_gate(hidden)
        return hidden * gate, gate

    @staticmethod
    def _segmented_prefix_cumsum(values, sorted_keys):
        """Inclusive prefix sums without subtracting large group offsets."""
        result = values
        offset = 1
        while offset < sorted_keys.numel():
            same_group = sorted_keys[offset:] == sorted_keys[:-offset]
            shape = (same_group.size(0),) + (1,) * (values.ndim - 1)
            tail = result[offset:] + result[:-offset] * same_group.view(shape)
            result = torch.cat([result[:offset], tail], dim=0)
            offset *= 2
        return result

    @staticmethod
    def _segmented_prefix_stats(
        values, keys, valid, decay=None, occurrence_weighting=False
    ):
        flat_values = values.reshape(-1, values.size(-1))
        flat_keys = keys.reshape(-1)
        flat_valid = valid.reshape(-1).to(values.dtype)
        order = torch.argsort(flat_keys, stable=True)
        sorted_valid = flat_valid[order]
        sorted_keys = flat_keys[order]

        starts = torch.ones_like(sorted_keys, dtype=torch.bool)
        starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_ids = starts.cumsum(dim=0) - 1
        start_indices = torch.nonzero(starts, as_tuple=False).squeeze(-1)

        segmented_count = CausalTargetTransitionGraph._segmented_prefix_cumsum(
            sorted_valid, sorted_keys
        )

        if decay is None:
            sorted_weight = sorted_valid
            weighted_values = flat_values[order]
        else:
            if occurrence_weighting:
                positions = (
                    segmented_count.to(torch.float64) - 1.0
                ).clamp_min(0.0)
            else:
                positions = torch.arange(
                    keys.size(-1), device=keys.device, dtype=torch.float64
                ).view(1, -1).expand_as(keys).reshape(-1)[order]
            ends = torch.cat([
                start_indices[1:] - 1,
                start_indices.new_tensor([sorted_keys.numel() - 1]),
            ])
            group_max_position = positions[ends]
            stable_exponent = group_max_position[group_ids] - positions
            # Multiplying all weights in a group by the same constant leaves
            # every prefix mean unchanged.  Keeping exponents non-negative
            # avoids the exponential growth that appears on long histories.
            sorted_weight = sorted_valid.to(torch.float64) * torch.pow(
                positions.new_tensor(float(decay)), stable_exponent
            )
            weighted_values = flat_values[order].to(torch.float64)
        sorted_values = (
            weighted_values * sorted_weight.unsqueeze(-1)
        )
        segmented_values = CausalTargetTransitionGraph._segmented_prefix_cumsum(
            sorted_values, sorted_keys
        )
        segmented_weight = CausalTargetTransitionGraph._segmented_prefix_cumsum(
            sorted_weight, sorted_keys
        )
        sorted_mean = (
            segmented_values
            / segmented_weight.clamp_min(1e-12).unsqueeze(-1)
        )

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        mean = sorted_mean[inverse].view_as(values).to(values.dtype)
        count = segmented_count[inverse].view(*keys.shape, 1)
        return mean, count

    @staticmethod
    def _segmented_prefix_mean(values, keys, valid):
        return CausalTargetTransitionGraph._segmented_prefix_stats(
            values, keys, valid
        )[0]

    def forward_sequence(
        self,
        concept_seq,
        seq_len,
        concept_states,
        outcome_states,
        edge_valid_mask=None,
        transition_keys=None,
        item_states=None,
    ):
        steps = max(concept_seq.size(1) - 1, 0)
        if steps == 0:
            empty_state = concept_states.new_zeros(
                concept_seq.size(0), 0, concept_states.size(-1)
            )
            return empty_state, empty_state[..., :1]

        source_ids = concept_seq[:, :-1]
        target_ids = concept_seq[:, 1:]
        if item_states is None:
            if concept_seq.ndim != 2:
                raise ValueError(
                    'multi-concept transitions require pooled item_states'
                )
            source_states = concept_states[source_ids]
            target_states = concept_states[target_ids]
        else:
            if item_states.shape[:2] != concept_seq.shape[:2]:
                raise ValueError('transition item_states shape mismatch')
            source_states = item_states[:, :-1]
            target_states = item_states[:, 1:]
        messages, gates = self._edge_states(
            source_states,
            target_states,
            outcome_states,
        )
        positions = torch.arange(steps, device=concept_seq.device)
        edge_valid = positions.view(1, -1) < (seq_len - 1).unsqueeze(1)
        if edge_valid_mask is not None:
            if edge_valid_mask.shape != edge_valid.shape:
                raise ValueError('transition edge mask shape mismatch')
            edge_valid = edge_valid & edge_valid_mask.bool()
        key_seq = concept_seq if transition_keys is None else transition_keys
        if key_seq.ndim != 2 or key_seq.shape != concept_seq.shape[:2]:
            raise ValueError(
                'transition_keys must have shape [batch, sequence]'
            )
        target_keys = key_seq[:, 1:]
        key_span = max(int(target_keys.max().item()) + 1, 1)
        batch_offset = (
            torch.arange(concept_seq.size(0), device=concept_seq.device)
            * key_span
        ).unsqueeze(1)
        context, count = self._segmented_prefix_stats(
            messages,
            batch_offset + target_keys,
            edge_valid,
            decay=self.decay if self.aggregation != 'uniform' else None,
            occurrence_weighting=(
                self.aggregation == 'occurrence_recency'
            ),
        )
        state = self.output_norm(self.output_projection(context))
        reliability = count / (count + self.reliability_strength)
        state = state.masked_fill(~edge_valid.unsqueeze(-1), 0.0)
        reliability = reliability.masked_fill(~edge_valid.unsqueeze(-1), 0.0)
        self.last_edge_gate = gates.detach()
        return state, reliability

    def forward_target(
        self,
        concept_seq,
        target_ids,
        seq_len,
        concept_states,
        outcome_states,
        edge_valid_mask=None,
    ):
        batch_size, steps = concept_seq.shape
        destination_ids = torch.roll(concept_seq, shifts=-1, dims=1)
        last = (seq_len - 1).clamp_min(0)
        destination_ids = destination_ids.scatter(
            1, last.unsqueeze(1), target_ids.unsqueeze(1)
        )
        messages, gates = self._edge_states(
            concept_states[concept_seq],
            concept_states[destination_ids],
            outcome_states,
        )
        positions = torch.arange(steps, device=concept_seq.device).view(1, -1)
        valid = positions < seq_len.unsqueeze(1)
        if edge_valid_mask is not None:
            if edge_valid_mask.shape != valid.shape:
                raise ValueError('transition edge mask shape mismatch')
            valid = valid & edge_valid_mask.bool()
        mask = valid & (destination_ids == target_ids.unsqueeze(1))
        count = mask.sum(dim=1, keepdim=True).to(messages.dtype)
        if self.aggregation == 'recency':
            relative_age = (
                (seq_len - 1).unsqueeze(1) - positions
            ).clamp_min(0).to(messages.dtype)
            weight = mask.to(messages.dtype) * torch.pow(
                messages.new_tensor(self.decay), relative_age
            )
        elif self.aggregation == 'occurrence_recency':
            rank = mask.cumsum(dim=1).to(messages.dtype) - 1.0
            relative_age = (count - 1.0 - rank).clamp_min(0.0)
            weight = mask.to(messages.dtype) * torch.pow(
                messages.new_tensor(self.decay), relative_age
            )
        else:
            weight = mask.to(messages.dtype)
        context = (
            messages * weight.unsqueeze(-1)
        ).sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
        state = self.output_norm(self.output_projection(context))
        reliability = count / (count + self.reliability_strength)
        self.last_edge_gate = gates.detach()
        return state * (count > 0).to(state.dtype), reliability


class EvidenceBranchAdapter(nn.Module):
    """Route one evidence state into a temporal branch before branch fusion."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, branch, target, evidence, reliability):
        reliability = reliability.to(branch.dtype).clamp(0.0, 1.0)
        gate = self.gate(torch.cat([
            branch, target, evidence, reliability
        ], dim=-1)) * reliability
        update = self.update(torch.cat([
            branch, target, evidence, branch * evidence
        ], dim=-1))
        return self.output_norm(branch + gate * update), gate


class BranchMemoryRouter(nn.Module):
    """Mix raw event and Mamba state memories before Transformer reading."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_gate = None

    def forward(self, events, states):
        gate = self.gate(torch.cat([
            events, states, events * states
        ], dim=-1))
        self.last_gate = gate.detach()
        return self.output_norm((1.0 - gate) * events + gate * states)


class ReliableEvidenceFusion(nn.Module):
    """Fuse any number of evidence modules through one shared gate."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, sequence, target, evidence):
        delta = torch.zeros_like(sequence)
        gates = {}
        for name, (state, reliability) in evidence.items():
            learned = self.gate(torch.cat([sequence, target, state], dim=-1))
            effective = learned * reliability.clamp(0.0, 1.0)
            delta = delta + effective * state
            gates[name] = effective
        if evidence:
            delta = delta / math.sqrt(len(evidence))
        fused = self.output_norm(
            sequence + self.dropout(self.projection(delta))
        )
        return fused, gates


class DGMKTStudentHypergraphProfile(nn.Module):
    """Reproduce DGMKT's ID-indexed student hypergraph profile."""

    def __init__(self, incidence, d_model):
        super().__init__()
        if incidence is None or incidence.ndim != 2:
            raise ValueError(
                'DGMKT student profile requires [students, concepts] incidence'
            )
        incidence = incidence.float()
        self.n_students = int(incidence.size(0))
        self.n_concepts = int(incidence.size(1))
        student_degree = incidence.sum(dim=1).clamp_min(1.0)
        concept_degree = incidence.sum(dim=0).clamp_min(1.0)
        self.register_buffer(
            'student_to_concept', incidence.to_sparse().coalesce()
        )
        self.register_buffer(
            'concept_to_student', incidence.T.to_sparse().coalesce()
        )
        self.register_buffer('student_scale', student_degree.rsqrt())
        self.register_buffer('concept_degree', concept_degree)

        # DGMKT creates an nn.Embedding, detaches it, and registers the result
        # as the fixed HGNN node input. nn.Embedding uses unit-normal anchors.
        anchor = torch.empty(self.n_students, d_model)
        nn.init.normal_(anchor)
        self.register_buffer('student_anchor', anchor)
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = nn.Parameter(torch.empty(d_model))
        bound = 1.0 / math.sqrt(d_model)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def _propagate(self, values):
        scaled = values * self.student_scale.unsqueeze(-1)
        concept = torch.sparse.mm(self.concept_to_student, scaled)
        concept = concept / self.concept_degree.unsqueeze(-1)
        student = torch.sparse.mm(self.student_to_concept, concept)
        return student * self.student_scale.unsqueeze(-1)

    def table(self):
        transformed = self.student_anchor.matmul(self.weight) + self.bias
        return F.relu(self._propagate(transformed))

    def forward(self, student_ids):
        # The released DGMKT forward pass uses one_hot(student_id - 1).
        student_ids = student_ids.long() - 1
        if (
            torch.any(student_ids < 0)
            or torch.any(student_ids >= self.n_students)
        ):
            raise ValueError('batch contains a student outside the DGMKT graph')
        return self.table()[student_ids]


class TargetAlignedHistoryAttention(nn.Module):
    """Augment recent Transformer state with prior events of the target concept."""

    def __init__(
        self,
        d_model,
        n_heads=4,
        max_matches=32,
        reliability_strength=3.0,
        dropout=0.2,
    ):
        super().__init__()
        if max_matches < 1:
            raise ValueError('target_history_size must be positive')
        self.max_matches = int(max_matches)
        self.reliability_strength = float(reliability_strength)
        self.layer = TargetCrossAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            window_size=self.max_matches,
            dropout=dropout,
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None
        self.last_gate = None
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def _select(self, memory, memory_ids, target_ids, valid_queries):
        batch_size, query_steps = target_ids.shape
        memory_steps = memory_ids.size(1)
        memory_positions = torch.arange(
            memory_steps, device=memory.device
        ).view(1, 1, -1)
        query_positions = torch.arange(
            query_steps, device=memory.device
        ).view(1, -1, 1)
        matches = (
            memory_ids.unsqueeze(1) == target_ids.unsqueeze(-1)
        )
        matches = matches & (memory_positions <= query_positions)
        matches = matches & valid_queries.unsqueeze(-1)
        position_scores = torch.where(
            matches,
            memory_positions.expand(batch_size, query_steps, -1),
            memory_positions.new_full(
                (batch_size, query_steps, memory_steps), -1
            ),
        )
        if memory_steps < self.max_matches:
            position_scores = F.pad(
                position_scores,
                (0, self.max_matches - memory_steps),
                value=-1,
            )
        selected_scores, selected = position_scores.topk(
            self.max_matches, dim=-1, largest=True, sorted=True
        )
        selected_mask = selected_scores >= 0
        selected = selected.clamp(max=max(memory_steps - 1, 0))
        batch_indices = torch.arange(
            batch_size, device=memory.device
        ).view(-1, 1, 1)
        selected_memory = memory[batch_indices, selected]
        counts = matches.sum(dim=-1, keepdim=True).to(memory.dtype)
        reliability = counts / (counts + self.reliability_strength)
        return selected_memory, selected_mask, reliability

    def _combine(self, recent, aligned, target, reliability):
        gate = torch.sigmoid(self.gate(torch.cat([
            recent,
            aligned,
            target,
            recent * aligned,
            reliability,
        ], dim=-1))) * reliability
        update = self.update(torch.cat([
            aligned,
            aligned * target,
        ], dim=-1))
        augmented = self.output_norm(recent + gate * update)
        output = torch.where(reliability > 0.0, augmented, recent)
        self.last_gate = gate.detach()
        return output, {
            'target_history': gate,
            'target_history_coverage': reliability,
        }

    def forward_sequence(
        self,
        history,
        concept_ids,
        seq_len,
        target_ids,
        target,
        recent,
    ):
        steps = target_ids.size(1)
        if steps == 0:
            empty = recent.new_zeros(*recent.shape[:-1], 1)
            return recent, {
                'target_history': empty,
                'target_history_coverage': empty,
            }
        memory = history[:, :steps]
        memory_ids = concept_ids[:, :steps]
        positions = torch.arange(steps, device=history.device)
        valid_queries = positions.unsqueeze(0) < (seq_len - 1).unsqueeze(1)
        selected, selected_mask, reliability = self._select(
            memory, memory_ids, target_ids, valid_queries
        )
        aligned, weights = self.layer(target, selected, selected_mask)
        self.last_attention_weights = weights.detach()
        output, diagnostics = self._combine(
            recent, aligned, target, reliability
        )
        return output.masked_fill(~valid_queries.unsqueeze(-1), 0.0), diagnostics

    def forward_target(
        self,
        history,
        concept_ids,
        seq_len,
        target_ids,
        target,
        recent,
    ):
        positions = torch.arange(
            history.size(1), device=history.device
        ).unsqueeze(0)
        valid_memory = positions < seq_len.unsqueeze(1)
        matches = (concept_ids == target_ids.unsqueeze(1)) & valid_memory
        position_scores = torch.where(
            matches,
            positions.expand_as(concept_ids),
            positions.new_full(concept_ids.shape, -1),
        )
        if history.size(1) < self.max_matches:
            position_scores = F.pad(
                position_scores,
                (0, self.max_matches - history.size(1)),
                value=-1,
            )
        selected_scores, selected = position_scores.topk(
            self.max_matches, dim=-1, largest=True, sorted=True
        )
        selected_mask = selected_scores >= 0
        selected = selected.clamp(max=max(history.size(1) - 1, 0))
        batch_indices = torch.arange(
            history.size(0), device=history.device
        ).unsqueeze(1)
        selected_memory = history[batch_indices, selected]
        counts = matches.sum(dim=-1, keepdim=True).to(history.dtype)
        reliability = counts / (counts + self.reliability_strength)
        aligned, weights = self.layer(
            target.unsqueeze(1),
            selected_memory.unsqueeze(1),
            selected_mask.unsqueeze(1),
        )
        self.last_attention_weights = weights.detach()
        output, diagnostics = self._combine(
            recent,
            aligned.squeeze(1),
            target,
            reliability,
        )
        return output, diagnostics


class LocalConditionedMambaInput(nn.Module):
    """Inject a causal target-local state before the Mamba transition."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.last_gate = None
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, events, local, target, valid=None):
        update = self.update(torch.cat([
            local,
            target,
            local * target,
        ], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([
            events,
            local,
            target,
            events * local,
            (local - target).abs(),
        ], dim=-1)))
        if valid is not None:
            gate = gate * valid.unsqueeze(-1).to(gate.dtype)
        self.last_gate = gate.detach()
        return events + gate * update, gate.mean(dim=-1, keepdim=True)


class StudentProfileConditioner(nn.Module):
    """Condition one temporal branch on the released student profile."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.last_gate = None
        # Start as an exact identity map so conditioning must earn its update.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, base, profile, target, valid=None):
        update = self.update(torch.cat([
            profile,
            target,
            profile * target,
        ], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([
            base,
            profile,
            target,
            base * profile,
            profile * target,
        ], dim=-1)))
        if valid is not None:
            gate = gate * valid.unsqueeze(-1).to(gate.dtype)
        self.last_gate = gate.detach()
        return base + gate * update, gate.mean(dim=-1, keepdim=True)


class DualBranchStateFusion(nn.Module):
    """Fuse long Mamba state with target-query Transformer state only."""

    DENOISED_VECTOR_MODES = {
        'denoised_adaptive_orthogonal_vector',
        'denoised_adaptive_raw_short_vector',
    }
    MODES = {
        'adaptive_novel',
        'adaptive_orthogonal',
        'adaptive_orthogonal_vector',
        'denoised_adaptive_orthogonal_vector',
        'denoised_adaptive_raw_short_vector',
        'adaptive_scalar',
        'adaptive_shared',
        'adaptive_vector',
        'orthogonal_innovation',
        'orthogonal_shared_only',
        'orthogonal_novel_only',
        'target_competitive',
        'gated_bilinear',
        'concat_mlp',
        'mean',
        'long_only',
        'short_only',
    }

    def __init__(self, d_model, mode='orthogonal_innovation', dropout=0.2):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f'branch fusion mode must be one of {sorted(self.MODES)}'
            )
        self.mode = mode
        self.long_norm = nn.LayerNorm(d_model)
        self.short_norm = nn.LayerNorm(d_model)
        self.gate_context = nn.Sequential(
            nn.Linear(5 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.orthogonal_gates = nn.Linear(d_model, 2)
        self.scalar_router = nn.Linear(d_model, 1)
        self.vector_router = nn.Linear(d_model, d_model)
        self.competitive_score = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.adaptive_update_gates = nn.Linear(d_model, 2)
        self.shared_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.novel_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        if mode in self.DENOISED_VECTOR_MODES:
            # Keep every shared parameter on the same RNG stream as the
            # incumbent while adding a candidate-only calibration layer.
            rng_state = torch.get_rng_state()
            self.innovation_relevance = nn.Linear(4 * d_model, d_model)
            torch.set_rng_state(rng_state)
        else:
            self.innovation_relevance = None
        self.gated_router = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.bilinear_update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.concat_update = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_diagnostics = None
        nn.init.zeros_(self.orthogonal_gates.weight)
        nn.init.constant_(self.orthogonal_gates.bias, -1.0)
        nn.init.zeros_(self.scalar_router.weight)
        nn.init.zeros_(self.scalar_router.bias)
        nn.init.zeros_(self.vector_router.weight)
        nn.init.zeros_(self.vector_router.bias)
        nn.init.zeros_(self.competitive_score[-1].weight)
        nn.init.zeros_(self.competitive_score[-1].bias)
        nn.init.zeros_(self.adaptive_update_gates.weight)
        nn.init.constant_(self.adaptive_update_gates.bias, -2.0)
        if self.innovation_relevance is not None:
            nn.init.zeros_(self.innovation_relevance.weight)
            nn.init.zeros_(self.innovation_relevance.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        adaptive_optional = (
            'scalar_router.',
            'vector_router.',
            'competitive_score.',
            'adaptive_update_gates.',
        )
        legacy_mode = not (
            self.mode.startswith('adaptive_')
            or self.mode in {
                *self.DENOISED_VECTOR_MODES,
                'target_competitive',
            }
        )
        for name, value in self.state_dict().items():
            if legacy_mode and any(
                name.startswith(module)
                for module in adaptive_optional
            ):
                state_dict.setdefault(prefix + name, value.detach())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if legacy_mode:
            missing_keys[:] = [
                key for key in missing_keys
                if not any(
                    key.startswith(prefix + module)
                    for module in adaptive_optional
                )
            ]

    def _orthogonal_components(self, long_state, short_state):
        long_state = self.long_norm(long_state)
        short_state = self.short_norm(short_state)
        long_direction = F.normalize(long_state, dim=-1, eps=1e-6)
        overlap = (
            short_state * long_direction
        ).sum(dim=-1, keepdim=True) * long_direction
        novel = short_state - overlap
        novelty = novel.norm(dim=-1, keepdim=True) / (
            short_state.norm(dim=-1, keepdim=True) + 1e-6
        )
        return long_state, short_state, overlap, novel, novelty

    @staticmethod
    def _interaction_inputs(long_state, short_state, target):
        return torch.cat([
            long_state,
            short_state,
            target,
            long_state * short_state,
            (long_state - short_state).abs(),
        ], dim=-1)

    def _competitive_weights(self, long_state, short_state, target):
        def score(branch):
            return self.competitive_score(torch.cat([
                branch,
                target,
                branch * target,
                (branch - target).abs(),
            ], dim=-1))

        scores = torch.cat([
            score(long_state), score(short_state)
        ], dim=-1)
        return torch.softmax(scores, dim=-1)

    def forward(self, long_state, short_state, target):
        if self.mode == 'mean':
            long_state = self.long_norm(long_state)
            short_state = self.short_norm(short_state)
            diagnostic_shape = (*short_state.shape[:-1], 1)
            zeros = short_state.new_zeros(diagnostic_shape)
            diagnostics = {
                'branch_short_weight': short_state.new_full(
                    diagnostic_shape, 0.5
                ),
                'branch_shared': zeros,
                'branch_innovation': zeros,
                'branch_novelty': zeros,
            }
            self.last_diagnostics = {
                name: value.detach() for name, value in diagnostics.items()
            }
            fused = 0.5 * (long_state + short_state)
            return self.output_norm(fused), diagnostics

        if self.mode == 'denoised_adaptive_raw_short_vector':
            long_state = self.long_norm(long_state)
            short_state = self.short_norm(short_state)
            overlap = torch.zeros_like(short_state)
            novel = short_state
            novelty = short_state.new_ones(*short_state.shape[:-1], 1)
        else:
            long_state, short_state, overlap, novel, novelty = (
                self._orthogonal_components(long_state, short_state)
            )
        zeros = novelty.new_zeros(novelty.shape)
        short_weight = zeros

        if self.mode == 'long_only':
            fused = long_state
            shared_gate, novel_gate = zeros, zeros
        elif self.mode == 'short_only':
            fused = short_state
            shared_gate, novel_gate = zeros, novelty
            short_weight = novelty.new_ones(novelty.shape)
        elif self.mode == 'gated_bilinear':
            inputs = self._interaction_inputs(long_state, short_state, target)
            gate = self.gated_router(inputs)
            interaction = self.bilinear_update(long_state * short_state)
            fused = gate * long_state + (1.0 - gate) * short_state + interaction
            shared_gate = gate.mean(dim=-1, keepdim=True)
            novel_gate = 1.0 - shared_gate
            short_weight = novel_gate
        elif self.mode == 'concat_mlp':
            inputs = self._interaction_inputs(long_state, short_state, target)
            fused = long_state + self.concat_update(inputs)
            shared_gate = novel_gate = novelty.new_full(novelty.shape, 1.0)
        elif (
            self.mode.startswith('adaptive_')
            or self.mode in self.DENOISED_VECTOR_MODES
            or self.mode == 'target_competitive'
        ):
            context = self.gate_context(torch.cat([
                long_state,
                short_state,
                target,
                long_state * short_state,
                novel,
                novelty,
            ], dim=-1))
            if self.mode == 'adaptive_vector':
                mix = torch.sigmoid(self.vector_router(context))
            elif self.mode in {
                'adaptive_orthogonal_vector',
                *self.DENOISED_VECTOR_MODES,
            }:
                mix = torch.sigmoid(self.vector_router(context))
            elif self.mode == 'target_competitive':
                weights = self._competitive_weights(
                    long_state, short_state, target
                )
                mix = weights[..., 1:2]
            else:
                mix = torch.sigmoid(self.scalar_router(context))
            fused = (1.0 - mix) * long_state + mix * short_state
            short_weight = mix.mean(dim=-1, keepdim=True)
            shared_gate = novel_gate = zeros

            if self.mode in {
                'adaptive_shared',
                'adaptive_novel',
                'adaptive_orthogonal',
                'adaptive_orthogonal_vector',
                *self.DENOISED_VECTOR_MODES,
            }:
                update_gates = torch.sigmoid(
                    self.adaptive_update_gates(context)
                )
                shared_gate = update_gates[..., :1]
                novel_gate = update_gates[..., 1:]
                if self.mode == 'adaptive_shared':
                    novel_gate = torch.zeros_like(novel_gate)
                elif self.mode == 'adaptive_novel':
                    shared_gate = torch.zeros_like(shared_gate)
                shared = self.shared_update(torch.cat([
                    long_state * short_state,
                    target,
                ], dim=-1))
                innovation = self.novel_update(torch.cat([
                    novel,
                    novel * target,
                ], dim=-1))
                if self.mode in self.DENOISED_VECTOR_MODES:
                    relevance = 1.0 + torch.tanh(
                        self.innovation_relevance(torch.cat([
                            novel,
                            target,
                            novel * target,
                            (novel - target).abs(),
                        ], dim=-1))
                    )
                    shared_base = (
                        (1.0 - mix) * long_state + mix * overlap
                    )
                    innovation_path = (
                        mix * novel
                        + novel_gate * novelty * innovation
                    )
                    fused = (
                        shared_base
                        + shared_gate * shared
                        + relevance * innovation_path
                    )
                else:
                    relevance = torch.ones_like(fused)
                    fused = (
                        fused
                        + shared_gate * shared
                        + novel_gate * novelty * innovation
                    )
            else:
                relevance = torch.ones_like(fused)
        else:
            context = self.gate_context(torch.cat([
                long_state,
                short_state,
                target,
                long_state * short_state,
                novel,
                novelty,
            ], dim=-1))
            gates = torch.sigmoid(self.orthogonal_gates(context))
            shared_gate = gates[..., :1]
            novel_gate = gates[..., 1:]
            if self.mode == 'orthogonal_shared_only':
                novel_gate = torch.zeros_like(novel_gate)
            elif self.mode == 'orthogonal_novel_only':
                shared_gate = torch.zeros_like(shared_gate)
            shared = self.shared_update(torch.cat([
                long_state * short_state,
                target,
            ], dim=-1))
            innovation = self.novel_update(torch.cat([
                novel,
                novel * target,
            ], dim=-1))
            fused = (
                long_state
                + shared_gate * shared
                + novel_gate * novelty * innovation
            )
            relevance = torch.ones_like(fused)

        if not (
            self.mode.startswith('adaptive_')
            or self.mode in self.DENOISED_VECTOR_MODES
            or self.mode == 'target_competitive'
        ):
            relevance = torch.ones_like(fused)

        diagnostics = {
            'branch_short_weight': short_weight,
            'branch_shared': shared_gate,
            'branch_innovation': novel_gate,
            'branch_novelty': novelty,
        }
        if self.mode in self.DENOISED_VECTOR_MODES:
            diagnostics['branch_innovation_relevance'] = relevance.mean(
                dim=-1, keepdim=True
            )
        self.last_diagnostics = {
            name: value.detach() for name, value in diagnostics.items()
        }
        return self.output_norm(fused), diagnostics


class BaselineBranchFusion(nn.Module):
    """Fuse both temporal branches without any orthogonal-fusion machinery."""

    MODES = {
        'fixed_mean_no_orthogonal_fusion',
        'concat_linear_no_orthogonal_fusion',
        'normalized_channel_weighted_sum',
    }

    def __init__(self, d_model, mode):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f'baseline fusion mode must be one of {sorted(self.MODES)}'
            )
        self.mode = mode
        self.long_norm = nn.LayerNorm(d_model)
        self.short_norm = nn.LayerNorm(d_model)
        self.concat_projection = (
            nn.Linear(2 * d_model, d_model)
            if mode == 'concat_linear_no_orthogonal_fusion'
            else None
        )
        self.branch_logits = (
            nn.Parameter(torch.zeros(2, d_model))
            if mode == 'normalized_channel_weighted_sum'
            else None
        )
        self.weighted_projection = (
            nn.Linear(d_model, d_model)
            if mode == 'normalized_channel_weighted_sum'
            else None
        )
        if self.weighted_projection is not None:
            nn.init.eye_(self.weighted_projection.weight)
            nn.init.zeros_(self.weighted_projection.bias)
        self.output_norm = nn.LayerNorm(d_model)
        self.last_diagnostics = None

    def forward(self, long_state, short_state, target):
        del target
        long_state = self.long_norm(long_state)
        short_state = self.short_norm(short_state)
        if self.mode == 'fixed_mean_no_orthogonal_fusion':
            fused = 0.5 * (long_state + short_state)
            short_weight = None
        elif self.mode == 'concat_linear_no_orthogonal_fusion':
            fused = self.concat_projection(torch.cat([
                long_state, short_state,
            ], dim=-1))
            short_weight = None
        else:
            weights = torch.softmax(self.branch_logits, dim=0)
            fused = (
                weights[0] * long_state + weights[1] * short_state
            )
            fused = self.weighted_projection(fused)
            short_weight = weights[1].mean()

        diagnostic_shape = (*short_state.shape[:-1], 1)
        zeros = short_state.new_zeros(diagnostic_shape)
        diagnostic_short_weight = short_state.new_full(
            diagnostic_shape, 0.5
        )
        if short_weight is not None:
            diagnostic_short_weight = short_weight.expand(
                diagnostic_shape
            )
        diagnostics = {
            'branch_short_weight': diagnostic_short_weight,
            'branch_shared': zeros,
            'branch_innovation': zeros,
            'branch_novelty': zeros,
        }
        self.last_diagnostics = {
            name: value.detach() for name, value in diagnostics.items()
        }
        return self.output_norm(fused), diagnostics


class ModularEvidenceKT(nn.Module):
    """One sequence core, modular general evidence, and one prediction head."""

    def __init__(
        self,
        n_questions,
        n_concepts,
        d_model=128,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.2,
        task_mode='concept',
        mamba_version='mamba2',
        mamba_layers=2,
        temporal_backbone='mamba',
        n_heads=4,
        n_layers=2,
        short_window=64,
        summary_block_size=32,
        max_seq_len=500,
        short_memory_mode='contiguous',
        target_history_size=32,
        branch_fusion='orthogonal_innovation',
        short_memory_source='event',
        mamba_interaction='parallel',
        dynamics_mode='entangled',
        concept_graph=None,
        concept_statistics=None,
        use_difficulty=True,
        use_student_graph=False,
        student_incidence=None,
        use_population_graph=True,
        population_fusion='evidence',
        use_mastery=True,
        use_concept_memory=False,
        use_transition_graph=True,
        transition_aggregation='uniform',
        transition_decay=0.97,
        outcome_calibration='static',
        outcome_prior_strength=None,
        evidence_placement='posthoc',
        use_target_retrieval=False,
        retrieval_source='event',
        retrieval_scope='all',
        retrieval_heads=4,
        ability_mode='off',
        ability_refinement_steps=2,
        target_conditioned_readout=True,
        concept_memory_fusion='competitive',
        evidence_prior_strength=5.0,
        use_item_context=False,
        item_statistics=None,
        item_concept_incidence=None,
        attempt_grouped_transition=False,
        student_conditioning='off',
        use_question_context=False,
        question_graph=None,
        question_concept_incidence=None,
        concept_question_incidence=None,
        use_question_rasch=False,
        use_bundle_attention_bias=False,
        bundle_attention_strength=0.5,
        bundle_attention_decay=0.9,
    ):
        super().__init__()
        if str(task_mode).lower() != 'concept':
            raise ValueError('V8 currently implements the shared concept-level contract')
        if n_concepts < 2:
            raise ValueError('n_concepts must include padding and at least one concept')
        if concept_graph is None:
            concept_graph = torch.eye(n_concepts)
        if tuple(concept_graph.shape) != (n_concepts, n_concepts):
            raise ValueError('concept graph size must match n_concepts')

        self.use_difficulty = bool(use_difficulty)
        self.use_student_graph = bool(use_student_graph)
        self.use_question_context = bool(use_question_context)
        self.use_question_rasch = bool(use_question_rasch)
        self.use_bundle_attention_bias = bool(use_bundle_attention_bias)
        if self.use_question_context and n_questions < 2:
            raise ValueError(
                'question context requires a question vocabulary'
            )
        if self.use_question_rasch and not self.use_question_context:
            raise ValueError('question Rasch residual requires question context')
        if temporal_backbone not in {'mamba', 'transformer', 'mamba_transformer'}:
            raise ValueError(
                'temporal_backbone must be mamba, transformer, or '
                'mamba_transformer'
            )
        self.temporal_backbone = temporal_backbone
        self.has_mamba = temporal_backbone in {'mamba', 'mamba_transformer'}
        self.has_transformer = temporal_backbone in {
            'transformer', 'mamba_transformer'
        }
        if self.use_bundle_attention_bias and not self.has_transformer:
            raise ValueError(
                'bundle attention bias requires the Transformer branch'
            )
        if student_conditioning not in {'off', 'mamba', 'transformer', 'both'}:
            raise ValueError(
                'student_conditioning must be off, mamba, transformer, or both'
            )
        if student_conditioning != 'off' and not self.use_student_graph:
            raise ValueError('student conditioning requires student graph')
        if (
            student_conditioning in {'transformer', 'both'}
            and not self.has_transformer
        ):
            raise ValueError(
                'Transformer student conditioning requires a Transformer branch'
            )
        if student_conditioning in {'mamba', 'both'} and not self.has_mamba:
            raise ValueError('Mamba student conditioning requires a Mamba branch')
        if student_conditioning != 'off' and dynamics_mode != 'entangled':
            raise ValueError('student conditioning requires entangled dynamics')
        self.student_conditioning = student_conditioning
        self.item_context_requested = bool(use_item_context)
        item_context_available = bool(
            item_statistics is not None
            and item_statistics.ndim == 2
            and item_statistics.size(1) >= 2
            and torch.any(item_statistics[:, 1] > 0).item()
        )
        self.use_item_context = (
            self.item_context_requested and item_context_available
        )
        self.attempt_grouped_transition = bool(attempt_grouped_transition)
        if self.use_item_context and n_questions < 2:
            raise ValueError('item context requires a question vocabulary')
        if short_memory_source not in {'event', 'state', 'mean', 'gated'}:
            raise ValueError(
                'short_memory_source must be event, state, mean, or gated'
            )
        if (
            short_memory_source != 'event'
            and temporal_backbone != 'mamba_transformer'
        ):
            raise ValueError(
                'non-event short memory requires mamba_transformer'
            )
        self.short_memory_source = short_memory_source
        if mamba_interaction not in {
            'parallel',
            'local_to_mamba',
            'local_to_mamba_fused',
            'interleaved_fused',
        }:
            raise ValueError(
                'mamba_interaction must be parallel, local_to_mamba, or '
                'local_to_mamba_fused, or interleaved_fused'
            )
        if (
            mamba_interaction != 'parallel'
            and temporal_backbone != 'mamba_transformer'
        ):
            raise ValueError(
                'Mamba-Transformer interaction requires mamba_transformer'
            )
        if (
            mamba_interaction in {'local_to_mamba', 'local_to_mamba_fused'}
            and short_memory_source != 'event'
        ):
            raise ValueError(
                'local-to-Mamba interaction requires event short memory to '
                'avoid a circular Mamba-to-Transformer dependency'
            )
        if mamba_interaction == 'interleaved_fused' and mamba_layers < 2:
            raise ValueError(
                'interleaved Mamba-Transformer interaction requires at '
                'least two Mamba layers'
            )
        self.mamba_interaction = mamba_interaction
        if short_memory_mode not in {'contiguous', 'contiguous_target'}:
            raise ValueError(
                'short_memory_mode must be contiguous or contiguous_target'
            )
        self.short_memory_mode = short_memory_mode
        if (
            self.temporal_backbone in {'transformer', 'mamba_transformer'}
            and dynamics_mode != 'entangled'
        ):
            raise ValueError(
                'Transformer backbones require entangled dynamics'
            )
        if dynamics_mode not in {'entangled', 'factorized'}:
            raise ValueError(
                'dynamics_mode must be entangled or factorized'
            )
        self.dynamics_mode = dynamics_mode
        self.use_population_graph = bool(use_population_graph)
        if population_fusion not in {'evidence', 'backbone', 'both'}:
            raise ValueError(
                'population_fusion must be evidence, backbone, or both'
            )
        self.population_fusion = population_fusion
        self.use_mastery = bool(use_mastery)
        self.use_concept_memory = bool(use_concept_memory)
        self.use_transition_graph = bool(use_transition_graph)
        if transition_aggregation not in {
            'uniform', 'recency', 'occurrence_recency'
        }:
            raise ValueError(
                'transition_aggregation must be uniform, recency, or '
                'occurrence_recency'
            )
        self.transition_aggregation = transition_aggregation
        self.transition_decay = float(transition_decay)
        self.evidence_prior_strength = float(evidence_prior_strength)
        self.outcome_prior_strength = float(
            evidence_prior_strength
            if outcome_prior_strength is None else outcome_prior_strength
        )
        if self.outcome_prior_strength <= 0.0:
            raise ValueError('outcome_prior_strength must be positive')
        if outcome_calibration not in {'static', 'prefix_posterior'}:
            raise ValueError(
                'outcome_calibration must be static or prefix_posterior'
            )
        self.outcome_calibration = outcome_calibration
        if evidence_placement not in {
            'posthoc',
            'student_long',
            'transition_short',
            'semantic_split',
            'student_long_both',
            'transition_short_both',
            'semantic_split_both',
        }:
            raise ValueError('unsupported evidence_placement')
        if (
            evidence_placement != 'posthoc'
            and temporal_backbone != 'mamba_transformer'
        ):
            raise ValueError(
                'branch evidence placement requires mamba_transformer'
            )
        if (
            evidence_placement in {
                'student_long',
                'semantic_split',
                'student_long_both',
                'semantic_split_both',
            }
            and not self.use_student_graph
        ):
            raise ValueError('student branch placement requires student graph')
        if (
            evidence_placement in {
                'transition_short',
                'semantic_split',
                'transition_short_both',
                'semantic_split_both',
            }
            and not self.use_transition_graph
        ):
            raise ValueError(
                'transition branch placement requires transition graph'
            )
        self.evidence_placement = evidence_placement
        self.keep_routed_posthoc = evidence_placement.endswith('_both')
        self.use_target_retrieval = bool(use_target_retrieval)
        if retrieval_source not in {'event', 'state', 'hybrid'}:
            raise ValueError(
                'retrieval_source must be event, state, or hybrid'
            )
        self.retrieval_source = retrieval_source
        if retrieval_scope not in {'all', 'same', 'adaptive'}:
            raise ValueError(
                'retrieval_scope must be all, same, or adaptive'
            )
        self.retrieval_scope = retrieval_scope
        if ability_mode not in {'off', 'evidence', 'innovation', 'both'}:
            raise ValueError(
                'ability_mode must be off, evidence, innovation, or both'
            )
        self.ability_mode = ability_mode
        self.target_conditioned_readout = bool(target_conditioned_readout)
        if concept_memory_fusion not in {'residual', 'competitive'}:
            raise ValueError(
                'concept_memory_fusion must be residual or competitive'
            )
        self.concept_memory_fusion = concept_memory_fusion
        self.concept_encoder = DifficultyAwareConceptEncoder(
            n_concepts,
            d_model,
            concept_statistics=concept_statistics,
            dropout=dropout,
            enabled=self.use_difficulty,
        )
        self.population_encoder = PopulationConceptHypergraph(
            concept_graph,
            d_model=d_model,
            dropout=dropout,
            enabled=self.use_population_graph,
        )
        self.population_backbone_adapter = PopulationBackboneAdapter(
            d_model,
            dropout=dropout,
        )
        self.population_evidence = PrefixPopulationEvidence(
            d_model,
            reliability_strength=evidence_prior_strength,
            dropout=dropout,
        )
        self.mastery_evidence = TargetMasteryEvidence(
            d_model,
            prior_strength=evidence_prior_strength,
            dropout=dropout,
        )
        self.concept_memory = DynamicConceptMemory(
            d_model,
            prior_strength=evidence_prior_strength,
            dropout=dropout,
        )
        self.coverage_fusion = CoverageAdaptiveFusion(
            d_model,
            dropout=dropout,
        )
        self.transition_evidence = CausalTargetTransitionGraph(
            d_model,
            reliability_strength=evidence_prior_strength,
            dropout=dropout,
            aggregation=self.transition_aggregation,
            decay=self.transition_decay,
        )
        self.target_retrieval = TargetConditionedCausalRetrieval(
            d_model,
            n_heads=retrieval_heads,
            reliability_strength=evidence_prior_strength,
            dropout=dropout,
        )
        self.history_router = CoverageAdaptiveHistoryRouter(
            d_model,
            dropout=dropout,
        )
        self.rasch_ability = CausalRaschAbility(
            d_model,
            prior_precision=1.0,
            reliability_strength=evidence_prior_strength,
            refinement_steps=ability_refinement_steps,
            dropout=dropout,
        )
        self.rasch_innovation_adapter = RaschInnovationAdapter(
            d_model,
            dropout=dropout,
        )
        self.retrieval_memory_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.question_encoder = (
            StaticQuestionGraphEncoder(
                n_questions,
                n_concepts,
                d_model,
                question_graph,
                question_concept_incidence,
                concept_question_incidence=concept_question_incidence,
                use_question_rasch=self.use_question_rasch,
                dropout=dropout,
            )
            if self.use_question_context else None
        )
        self.question_concept_fusion = (
            nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            if self.use_question_context else None
        )

        self.response_embedding = nn.Embedding(2, d_model)
        self.outcome_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        encoder_kwargs = {
            'd_model': d_model,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'n_layers': mamba_layers,
            'dropout': dropout,
            'version': mamba_version,
        }
        if self.dynamics_mode == 'entangled' and self.has_mamba:
            self.sequence_encoder = ResidualMambaEncoder(**encoder_kwargs)
            self.exposure_sequence_encoder = None
            self.acquisition_sequence_encoder = None
        elif self.dynamics_mode == 'factorized':
            self.sequence_encoder = None
            self.exposure_sequence_encoder = ResidualMambaEncoder(
                **encoder_kwargs
            )
            self.acquisition_sequence_encoder = ResidualMambaEncoder(
                **encoder_kwargs
            )
        else:
            # A hard Transformer-only ablation must not instantiate Mamba.
            self.sequence_encoder = None
            self.exposure_sequence_encoder = None
            self.acquisition_sequence_encoder = None
        self.short_term = (
            TargetCrossAttentionTransformer(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                window_size=short_window,
                summary_block_size=summary_block_size,
                max_seq_len=max_seq_len,
                use_semantic_bias=self.use_bundle_attention_bias,
                semantic_bias_strength=bundle_attention_strength,
                semantic_bias_decay=bundle_attention_decay,
            )
            if self.has_transformer
            else None
        )
        if self.temporal_backbone != 'mamba_transformer':
            self.branch_fusion = None
        elif branch_fusion in BaselineBranchFusion.MODES:
            # Keep downstream initialization aligned with the full model while
            # physically excluding every parameter of its fusion module.
            rng_before_fusion = torch.get_rng_state()
            unused_full_fusion = DualBranchStateFusion(
                d_model=d_model,
                mode='denoised_adaptive_orthogonal_vector',
                dropout=dropout,
            )
            rng_after_full_fusion = torch.get_rng_state()
            del unused_full_fusion
            torch.set_rng_state(rng_before_fusion)
            self.branch_fusion = BaselineBranchFusion(
                d_model=d_model,
                mode=branch_fusion,
            )
            torch.set_rng_state(rng_after_full_fusion)
        else:
            self.branch_fusion = DualBranchStateFusion(
                d_model=d_model,
                mode=branch_fusion,
                dropout=dropout,
            )
        self.exposure_event_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.factorized_fusion = TargetConditionedFactorizedFusion(
            d_model,
            dropout=dropout,
        )
        self.factorized_memory_norm = nn.LayerNorm(d_model)
        self.target_readout_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.evidence_fusion = ReliableEvidenceFusion(d_model, dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Optional memory is initialized last so shared backbone, fusion, and
        # predictor weights remain identical to the contiguous reference.
        self.target_history = (
            TargetAlignedHistoryAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_matches=target_history_size,
                dropout=dropout,
            )
            if (
                self.has_transformer
                and self.short_memory_mode == 'contiguous_target'
            )
            else None
        )
        # Keep the DGMKT ID profile after all shared modules so enabling it does
        # not shift initialization of the fixed dual-branch backbone or head.
        self.student_graph_encoder = (
            DGMKTStudentHypergraphProfile(student_incidence, d_model)
            if self.use_student_graph else None
        )

        # Placement adapters are initialized after every shared module. Thus
        # the posthoc control is checkpoint-compatible and shared weights use
        # identical seeded initialization across placement variants.
        self.student_long_adapter = (
            EvidenceBranchAdapter(d_model, dropout=dropout)
            if self.evidence_placement in {
                'student_long',
                'semantic_split',
                'student_long_both',
                'semantic_split_both',
            }
            else None
        )
        self.transition_short_adapter = (
            EvidenceBranchAdapter(d_model, dropout=dropout)
            if self.evidence_placement in {
                'transition_short',
                'semantic_split',
                'transition_short_both',
                'semantic_split_both',
            }
            else None
        )
        self.branch_memory_router = (
            BranchMemoryRouter(d_model, dropout=dropout)
            if self.short_memory_source == 'gated' else None
        )
        self.mamba_input_conditioner = (
            LocalConditionedMambaInput(d_model, dropout=dropout)
            if self.mamba_interaction != 'parallel' else None
        )
        # Item evidence is initialized after every shared backbone/head module,
        # preserving identical seeded initialization for the control variant.
        self.item_evidence = (
            HierarchicalItemEvidence(
                n_questions,
                n_concepts,
                d_model,
                item_statistics,
                item_concept_incidence,
                dropout=dropout,
            )
            if self.use_item_context else None
        )
        # Initialize new profile conditioners last so every pre-existing module
        # keeps identical seeded initialization across round-13 variants.
        self.student_mamba_conditioner = (
            StudentProfileConditioner(d_model, dropout=dropout)
            if self.student_conditioning in {'mamba', 'both'} else None
        )
        self.student_transformer_conditioner = (
            StudentProfileConditioner(d_model, dropout=dropout)
            if self.student_conditioning in {'transformer', 'both'} else None
        )

    def _sequence_transition_mask(self, batch):
        if not self.attempt_grouped_transition:
            return None
        attempt_starts = batch.get('attempt_start_seq')
        if attempt_starts is None:
            raise ValueError(
                'attempt-grouped transition requires attempt_start_seq'
            )
        return attempt_starts[:, 1:]

    def _target_transition_mask(self, batch):
        if not self.attempt_grouped_transition:
            return None
        attempt_starts = batch.get('attempt_start_seq')
        if attempt_starts is None:
            raise ValueError(
                'attempt-grouped transition requires attempt_start_seq'
            )
        mask = torch.roll(attempt_starts, shifts=-1, dims=1)
        last = (batch['seq_len'] - 1).clamp_min(0)
        external_start = torch.ones_like(last, dtype=torch.bool)
        if batch.get('target_question') is not None:
            last_question = batch['question_seq'].gather(
                1, last.unsqueeze(1)
            ).squeeze(1)
            external_start = batch['target_question'] != last_question
        return mask.scatter(1, last.unsqueeze(1), external_start.unsqueeze(1))

    def _outcome_states(
        self,
        concepts,
        responses,
        success,
        attempts=None,
        correct=None,
    ):
        baseline = self._success_for_ids(success, concepts)
        if self.outcome_calibration == 'prefix_posterior':
            if attempts is None or correct is None:
                raise ValueError(
                    'prefix posterior calibration requires prefix counts'
                )
            prior = self.outcome_prior_strength
            baseline = (
                correct.to(baseline.dtype) + prior * baseline
            ) / (attempts.to(baseline.dtype) + prior)
        residual = responses.to(baseline.dtype) - baseline
        return self.outcome_encoder(residual.unsqueeze(-1))

    @staticmethod
    def _concept_states_for_ids(concept_states, concept_ids):
        """Look up one concept or mean-pool a padded concept bundle."""
        if concept_ids.ndim <= 2:
            return concept_states[concept_ids]
        values = concept_states[concept_ids]
        valid = concept_ids.ne(0).unsqueeze(-1).to(values.dtype)
        return (values * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)

    @staticmethod
    def _success_for_ids(success, concept_ids):
        """Look up success priors for scalar IDs or padded concept bundles."""
        if concept_ids.ndim <= 2:
            return success[concept_ids]
        values = success[concept_ids]
        valid = concept_ids.ne(0).to(values.dtype)
        return (values * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)

    def _event_context(
        self,
        batch,
        concept_states,
        success,
        event_attempts=None,
        event_correct=None,
        item_states=None,
    ):
        concepts = (
            self._concept_states_for_ids(
                concept_states, batch['concept_seq']
            )
            if item_states is None else item_states
        )
        responses = self.response_embedding(
            batch['response_seq'].long().clamp(0, 1)
        )
        outcome = self._outcome_states(
            batch['concept_seq'],
            batch['response_seq'],
            success,
            attempts=event_attempts,
            correct=event_correct,
        )
        return self.event_projection(torch.cat([
            concepts, responses, concepts * torch.tanh(responses), outcome
        ], dim=-1))

    def _sequence_item_states(self, batch, concept_states):
        concept_items = self._concept_states_for_ids(
            concept_states, batch['concept_seq']
        )
        if self.question_encoder is None:
            return concept_items
        question_seq = batch.get('question_seq')
        if question_seq is None:
            raise ValueError('question context requires question_seq')
        if question_seq.shape != batch['concept_seq'].shape[:2]:
            raise ValueError('question_seq shape mismatch')
        question_states = self.question_encoder(concept_states)[question_seq]
        return self.question_concept_fusion(torch.cat([
            concept_items,
            question_states,
            concept_items * question_states,
        ], dim=-1))

    @staticmethod
    def _event_prefix_counts(batch):
        provided_attempts = batch.get('event_concept_attempts')
        provided_correct = batch.get('event_concept_correct')
        if provided_attempts is not None and provided_correct is not None:
            return provided_attempts, provided_correct

        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None or correct is None:
            raise ValueError('prefix calibration requires initial concept counts')
        attempts = attempts.clone()
        correct = correct.clone()
        concepts = batch['concept_seq']
        responses = batch['response_seq']
        output_attempts = responses.new_zeros(responses.shape)
        output_correct = responses.new_zeros(responses.shape)
        indices = torch.arange(concepts.size(0), device=concepts.device)
        for position in range(concepts.size(1)):
            concept = concepts[:, position]
            output_attempts[:, position] = attempts[indices, concept]
            output_correct[:, position] = correct[indices, concept]
            valid = (position < batch['seq_len']).to(responses.dtype)
            attempts.scatter_add_(
                1, concept.unsqueeze(1), valid.unsqueeze(1)
            )
            correct.scatter_add_(
                1,
                concept.unsqueeze(1),
                (
                    valid * (responses[:, position] >= 0.5).to(responses.dtype)
                ).unsqueeze(1),
            )
        return output_attempts, output_correct

    @staticmethod
    def _sequence_target_counts(batch):
        provided_attempts = batch.get('target_concept_attempts')
        provided_correct = batch.get('target_concept_correct')
        if provided_attempts is not None and provided_correct is not None:
            return provided_attempts, provided_correct

        concepts = batch['concept_seq']
        responses = batch['response_seq']
        batch_size, length = concepts.shape
        n_concepts = batch.get('initial_concept_attempts').size(1)
        attempts = batch['initial_concept_attempts'].clone()
        correct = batch['initial_concept_correct'].clone()
        output_attempts = responses.new_zeros(batch_size, max(length - 1, 0))
        output_correct = responses.new_zeros(batch_size, max(length - 1, 0))
        indices = torch.arange(batch_size, device=concepts.device)
        for position in range(max(length - 1, 0)):
            valid = (position < batch['seq_len']).to(responses.dtype)
            source = concepts[:, position]
            attempts.scatter_add_(1, source.unsqueeze(1), valid.unsqueeze(1))
            correct.scatter_add_(
                1,
                source.unsqueeze(1),
                (valid * (responses[:, position] >= 0.5)).unsqueeze(1),
            )
            target = concepts[:, position + 1]
            output_attempts[:, position] = attempts[indices, target]
            output_correct[:, position] = correct[indices, target]
        return output_attempts, output_correct

    @staticmethod
    def _target_counts(batch):
        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None or correct is None:
            n_concepts = int(batch['concept_seq'].max().item()) + 1
            attempts = batch['response_seq'].new_zeros(
                batch['concept_seq'].size(0), n_concepts
            )
            correct = attempts.clone()
        else:
            attempts = attempts.clone()
            correct = correct.clone()
        positions = torch.arange(
            batch['concept_seq'].size(1), device=batch['concept_seq'].device
        )
        valid = positions.unsqueeze(0) < batch['seq_len'].unsqueeze(1)
        attempts.scatter_add_(
            1,
            batch['concept_seq'],
            valid.to(attempts.dtype),
        )
        correct.scatter_add_(
            1,
            batch['concept_seq'],
            valid.to(correct.dtype)
            * (batch['response_seq'] >= 0.5).to(correct.dtype),
        )
        indices = torch.arange(attempts.size(0), device=attempts.device)
        return (
            attempts[indices, batch['target_concept']],
            correct[indices, batch['target_concept']],
        )

    @staticmethod
    def _gather_last(values, seq_len):
        index = (seq_len - 1).clamp_min(0)
        batch = torch.arange(values.size(0), device=values.device)
        return values[batch, index]

    def _student_profile(self, batch):
        if self.student_graph_encoder is None:
            return None
        if batch.get('student_id') is None:
            raise ValueError('DGMKT student profile requires batch.student_id')
        return self.student_graph_encoder(batch['student_id'])

    def _condition_sequence(self, sequence, target):
        if not self.target_conditioned_readout:
            return sequence
        gate = self.target_readout_gate(
            torch.cat([sequence, target], dim=-1)
        )
        return sequence * (1.0 + gate)

    def _retrieval_memory(self, events, encoded):
        if self.retrieval_source == 'event':
            return events
        if self.retrieval_source == 'state':
            return encoded
        return self.retrieval_memory_projection(
            torch.cat([events, encoded], dim=-1)
        )

    def _factorized_states(self, batch, concept_states, events):
        exposure_events = self.exposure_event_projection(
            concept_states[batch['concept_seq']]
        )
        exposure = self.exposure_sequence_encoder(exposure_events)
        acquisition = self.acquisition_sequence_encoder(events)
        memory = self.factorized_memory_norm(exposure + acquisition)
        return exposure, acquisition, memory

    def _short_memory(self, events, encoded):
        if self.short_memory_source == 'event':
            return events, None
        if self.short_memory_source == 'state':
            return encoded, encoded.new_ones(*encoded.shape[:-1], 1)
        if self.short_memory_source == 'mean':
            return (
                (events + encoded) / math.sqrt(2.0),
                encoded.new_full((*encoded.shape[:-1], 1), 0.5),
            )
        memory = self.branch_memory_router(events, encoded)
        return memory, self.branch_memory_router.last_gate.mean(
            dim=-1, keepdim=True
        )

    def _mamba_layer_range(self, values, start, stop, finalize=False):
        encoder = self.sequence_encoder
        for index in range(start, stop):
            values = values + encoder.dropout(
                encoder.layers[index](encoder.norms[index](values))
            )
        return encoder.output_norm(values) if finalize else values

    def _concept_views(self):
        base = self.concept_encoder()
        population = None
        temporal = base
        if self.use_population_graph:
            population = self.population_encoder(base)
            if self.population_fusion in {'backbone', 'both'}:
                temporal = self.population_backbone_adapter(base, population)
        return base, temporal, population

    def _format_output(self, logits, fused, evidence, gates, return_aux):
        if not return_aux:
            return torch.sigmoid(logits)
        return {
            'logits': logits,
            'main_logits': logits,
            'fused_state': fused,
            'evidence_states': {
                name: state for name, (state, _) in evidence.items()
            },
            'evidence_reliability': {
                name: reliability for name, (_, reliability) in evidence.items()
            },
            'evidence_gates': gates,
        }

    def forward_sequence(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        _, concept_states, population_states = self._concept_views()
        item_states = self._sequence_item_states(batch, concept_states)
        success = self.concept_encoder.success_probability()
        event_attempts = event_correct = None
        if self.outcome_calibration == 'prefix_posterior':
            event_attempts, event_correct = self._event_prefix_counts(batch)
        ability = None
        if self.ability_mode != 'off':
            population_success = (
                1.0 - self.concept_encoder.difficulty
            ).clamp(1e-4, 1.0 - 1e-4)
            ability = self.rasch_ability.forward_sequence(
                batch, population_success
            )
        events = self._event_context(
            batch,
            concept_states,
            success,
            event_attempts=event_attempts,
            event_correct=event_correct,
            item_states=item_states,
        )
        if self.ability_mode in {'innovation', 'both'}:
            events = self.rasch_innovation_adapter(
                events, *ability['innovation']
            )
        target_ids = batch['concept_seq'][:, 1:]
        target = item_states[:, 1:]
        short_history_keys = batch.get('transition_key_seq')
        short_target_keys = (
            short_history_keys[:, 1:]
            if short_history_keys is not None else None
        )
        preliminary_gates = {}
        student_profile = self._student_profile(batch)
        short_target = target
        if self.student_transformer_conditioner is not None:
            profile = student_profile.unsqueeze(1).expand_as(target)
            positions = torch.arange(target.size(1), device=target.device)
            valid = positions.unsqueeze(0) < (
                batch['seq_len'] - 1
            ).unsqueeze(1)
            short_target, gate = self.student_transformer_conditioner(
                target, profile, target, valid=valid
            )
            preliminary_gates['student_transformer_conditioning'] = gate
        mamba_events = events
        if self.student_mamba_conditioner is not None:
            event_target = item_states
            profile = student_profile.unsqueeze(1).expand_as(events)
            positions = torch.arange(events.size(1), device=events.device)
            valid = positions.unsqueeze(0) < batch['seq_len'].unsqueeze(1)
            mamba_events, gate = self.student_mamba_conditioner(
                events, profile, event_target, valid=valid
            )
            preliminary_gates['student_mamba_conditioning'] = gate
        routed_student = None
        if self.student_long_adapter is not None:
            state = student_profile.unsqueeze(1).expand_as(target)
            routed_student = (
                state, state.new_ones(*state.shape[:-1], 1)
            )
        routed_transition = None
        if self.transition_short_adapter is not None:
            outcome = self._outcome_states(
                batch['concept_seq'][:, :-1],
                batch['response_seq'][:, :-1],
                success,
                attempts=(
                    event_attempts[:, :-1]
                    if event_attempts is not None else None
                ),
                correct=(
                    event_correct[:, :-1]
                    if event_correct is not None else None
                ),
            )
            routed_transition = self.transition_evidence.forward_sequence(
                batch['concept_seq'],
                batch['seq_len'],
                concept_states,
                outcome,
                edge_valid_mask=self._sequence_transition_mask(batch),
                transition_keys=batch.get('transition_key_seq'),
                item_states=item_states,
            )
        precomputed_short = None
        interleaved_encoded = None
        if self.mamba_interaction == 'interleaved_fused':
            first_stage = self._mamba_layer_range(
                mamba_events, 0, 1, finalize=False
            )
            short_memory, memory_gate = self._short_memory(
                events, first_stage
            )
            if memory_gate is not None:
                preliminary_gates['short_memory_state'] = (
                    memory_gate[:, :-1]
                )
            precomputed_short = self.short_term.forward_sequence(
                short_memory,
                batch['seq_len'],
                short_target,
                history_keys=short_history_keys,
                target_keys=short_target_keys,
            )
            if self.target_history is not None:
                precomputed_short, target_history_gates = (
                    self.target_history.forward_sequence(
                        short_memory,
                        batch['concept_seq'],
                        batch['seq_len'],
                        target_ids,
                        short_target,
                        precomputed_short,
                    )
                )
                preliminary_gates.update(target_history_gates)
            positions = torch.arange(
                precomputed_short.size(1), device=events.device
            )
            valid = positions.unsqueeze(0) < (
                batch['seq_len'] - 1
            ).unsqueeze(1)
            conditioned_prefix, conditioning_gate = (
                self.mamba_input_conditioner(
                    first_stage[:, :-1],
                    precomputed_short,
                    target,
                    valid=valid,
                )
            )
            second_stage_input = torch.cat([
                conditioned_prefix, first_stage[:, -1:]
            ], dim=1)
            interleaved_encoded = self._mamba_layer_range(
                second_stage_input,
                1,
                len(self.sequence_encoder.layers),
                finalize=True,
            )
            preliminary_gates['mamba_local_conditioning'] = conditioning_gate
            events_for_mamba = events
        elif self.mamba_input_conditioner is not None:
            precomputed_short = self.short_term.forward_sequence(
                events,
                batch['seq_len'],
                short_target,
                history_keys=short_history_keys,
                target_keys=short_target_keys,
            )
            if self.target_history is not None:
                precomputed_short, target_history_gates = (
                    self.target_history.forward_sequence(
                        events,
                        batch['concept_seq'],
                        batch['seq_len'],
                        target_ids,
                        short_target,
                        precomputed_short,
                    )
                )
                preliminary_gates.update(target_history_gates)
            positions = torch.arange(
                precomputed_short.size(1), device=events.device
            )
            valid = positions.unsqueeze(0) < (
                batch['seq_len'] - 1
            ).unsqueeze(1)
            conditioned_prefix, conditioning_gate = (
                self.mamba_input_conditioner(
                    mamba_events[:, :-1], precomputed_short, target, valid=valid
                )
            )
            events_for_mamba = torch.cat([
                conditioned_prefix, mamba_events[:, -1:]
            ], dim=1)
            preliminary_gates['mamba_local_conditioning'] = conditioning_gate
        else:
            events_for_mamba = mamba_events
        if self.dynamics_mode == 'factorized':
            exposure, acquisition, encoded = self._factorized_states(
                batch, concept_states, events
            )
            sequence = self.factorized_fusion(
                exposure[:, :-1], acquisition[:, :-1], target
            )
        else:
            # Transformer-only runs feed causal event states directly into the
            # short branch and never construct or execute a Mamba encoder.
            encoded = events
            long_state = None
            if self.has_mamba:
                encoded = (
                    interleaved_encoded
                    if interleaved_encoded is not None
                    else self.sequence_encoder(events_for_mamba)
                )
                long_state = self._condition_sequence(
                    encoded[:, :-1], target
                )
                if routed_student is not None:
                    long_state, preliminary_gates['student_long'] = (
                        self.student_long_adapter(
                            long_state, target, *routed_student
                        )
                    )
            if self.short_term is not None:
                if precomputed_short is None:
                    if self.has_mamba:
                        short_memory, memory_gate = self._short_memory(
                            events, encoded
                        )
                    else:
                        short_memory, memory_gate = events, None
                    if memory_gate is not None:
                        preliminary_gates['short_memory_state'] = (
                            memory_gate[:, :-1]
                        )
                    short_state = self.short_term.forward_sequence(
                        short_memory,
                        batch['seq_len'],
                        short_target,
                        history_keys=short_history_keys,
                        target_keys=short_target_keys,
                    )
                    if self.target_history is not None:
                        short_state, target_history_gates = (
                            self.target_history.forward_sequence(
                                short_memory,
                                batch['concept_seq'],
                                batch['seq_len'],
                                target_ids,
                                short_target,
                                short_state,
                            )
                        )
                        preliminary_gates.update(target_history_gates)
                else:
                    short_state = precomputed_short
                if routed_transition is not None:
                    short_state, preliminary_gates['transition_short'] = (
                        self.transition_short_adapter(
                            short_state, target, *routed_transition
                        )
                    )
                if not self.has_mamba:
                    sequence = short_state
                elif self.mamba_interaction == 'local_to_mamba':
                    sequence = long_state
                else:
                    sequence, branch_gates = self.branch_fusion(
                        long_state, short_state, target
                    )
                    preliminary_gates.update(branch_gates)
            else:
                sequence = long_state
        if self.dynamics_mode == 'factorized':
            sequence = self._condition_sequence(sequence, target)
        evidence = {}
        reported_evidence = {}

        if self.student_graph_encoder is not None:
            student_evidence = routed_student
            if student_evidence is None:
                student_state = student_profile.unsqueeze(1).expand_as(sequence)
                student_evidence = (
                    student_state,
                    student_state.new_ones(*student_state.shape[:-1], 1),
                )
            reported_evidence['student_graph'] = student_evidence
            if routed_student is None or self.keep_routed_posthoc:
                evidence['student_graph'] = student_evidence

        if self.ability_mode in {'evidence', 'both'}:
            evidence['ability'] = ability['evidence']

        if self.use_target_retrieval:
            memory = self._retrieval_memory(events, encoded)[:, :-1]
            memory_ids = batch['concept_seq'][:, :-1]
            if self.retrieval_scope == 'adaptive':
                broad = self.target_retrieval.forward_sequence(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope='all',
                )
                local = self.target_retrieval.forward_sequence(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope='same',
                )
                reported_evidence['broad_retrieval'] = broad
                reported_evidence['local_retrieval'] = local
                sequence, weights = self.history_router(
                    sequence, target, broad[0], local[0], local[1]
                )
                preliminary_gates.update({
                    'history_global': weights[..., 0:1],
                    'history_broad': weights[..., 1:2],
                    'history_local': weights[..., 2:3],
                })
            else:
                evidence['retrieval'] = self.target_retrieval.forward_sequence(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope=self.retrieval_scope,
                )

        if self.use_concept_memory:
            state, reliability = self.concept_memory.forward_sequence(
                batch, concept_states, success, events
            )
            reported_evidence['concept_memory'] = (state, reliability)
            if self.concept_memory_fusion == 'competitive':
                sequence, preliminary_gates['concept_memory'] = (
                    self.coverage_fusion(
                        sequence, target, state, reliability
                    )
                )
            else:
                evidence['concept_memory'] = (state, reliability)

        if self.use_item_context and self.item_evidence.available:
            question_seq = batch.get('question_seq')
            if question_seq is None:
                raise ValueError('item context requires question_seq')
            item_evidence = self.item_evidence(
                question_seq[:, 1:],
                target_ids,
                concept_states,
                success,
            )
            reported_evidence['item_hierarchy'] = item_evidence
            if torch.any(item_evidence[1] > 0):
                evidence['item_hierarchy'] = item_evidence

        if (
            self.use_population_graph
            and self.population_fusion in {'evidence', 'both'}
        ):
            state, reliability = self.population_evidence.forward_sequence(
                batch, population_states, success
            )
            evidence['population'] = (state[:, :-1], reliability[:, :-1])
        if self.use_mastery:
            attempts, correct = self._sequence_target_counts(batch)
            evidence['mastery'] = self.mastery_evidence(
                target_ids, attempts, correct, success
            )
        if self.use_transition_graph:
            transition_evidence = routed_transition
            if transition_evidence is None:
                outcome = self._outcome_states(
                    batch['concept_seq'][:, :-1],
                    batch['response_seq'][:, :-1],
                    success,
                    attempts=(
                        event_attempts[:, :-1]
                        if event_attempts is not None else None
                    ),
                    correct=(
                        event_correct[:, :-1]
                        if event_correct is not None else None
                    ),
                )
                transition_evidence = (
                    self.transition_evidence.forward_sequence(
                        batch['concept_seq'],
                        batch['seq_len'],
                        concept_states,
                        outcome,
                        edge_valid_mask=self._sequence_transition_mask(batch),
                        transition_keys=batch.get('transition_key_seq'),
                        item_states=item_states,
                    )
                )
            reported_evidence['transition'] = transition_evidence
            if routed_transition is None or self.keep_routed_posthoc:
                evidence['transition'] = transition_evidence

        fused, gates = self.evidence_fusion(sequence, target, evidence)
        gates = {**preliminary_gates, **gates}
        reported_evidence.update(evidence)
        logits = self.predictor(torch.cat([
            fused, target, fused * target
        ], dim=-1)).squeeze(-1)
        return self._format_output(
            logits, fused, reported_evidence, gates, return_aux
        )

    def forward(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        _, concept_states, population_states = self._concept_views()
        success = self.concept_encoder.success_probability()
        event_attempts = event_correct = None
        if self.outcome_calibration == 'prefix_posterior':
            event_attempts, event_correct = self._event_prefix_counts(batch)
        ability = None
        if self.ability_mode != 'off':
            population_success = (
                1.0 - self.concept_encoder.difficulty
            ).clamp(1e-4, 1.0 - 1e-4)
            ability = self.rasch_ability.forward_target(
                batch, population_success
            )
        events = self._event_context(
            batch,
            concept_states,
            success,
            event_attempts=event_attempts,
            event_correct=event_correct,
        )
        if self.ability_mode in {'innovation', 'both'}:
            events = self.rasch_innovation_adapter(
                events, *ability['innovation']
            )
        target_ids = batch['target_concept']
        target = concept_states[target_ids]
        short_history_keys = batch.get('transition_key_seq')
        short_target_keys = batch.get('target_transition_key')
        preliminary_gates = {}
        student_profile = self._student_profile(batch)
        short_target = target
        if self.student_transformer_conditioner is not None:
            short_target, gate = self.student_transformer_conditioner(
                target, student_profile, target
            )
            preliminary_gates['student_transformer_conditioning'] = gate
        mamba_events = events
        if self.student_mamba_conditioner is not None:
            event_target = concept_states[batch['concept_seq']]
            profile = student_profile.unsqueeze(1).expand_as(events)
            positions = torch.arange(
                events.size(1), device=events.device
            ).unsqueeze(0)
            valid = positions < batch['seq_len'].unsqueeze(1)
            mamba_events, gate = self.student_mamba_conditioner(
                events, profile, event_target, valid=valid
            )
            preliminary_gates['student_mamba_conditioning'] = gate
        routed_student = None
        if self.student_long_adapter is not None:
            routed_student = (
                student_profile,
                student_profile.new_ones(student_profile.size(0), 1),
            )
        routed_transition = None
        if self.transition_short_adapter is not None:
            outcome = self._outcome_states(
                batch['concept_seq'],
                batch['response_seq'],
                success,
                attempts=event_attempts,
                correct=event_correct,
            )
            routed_transition = self.transition_evidence.forward_target(
                batch['concept_seq'],
                target_ids,
                batch['seq_len'],
                concept_states,
                outcome,
                edge_valid_mask=self._target_transition_mask(batch),
            )
        precomputed_short = None
        interleaved_encoded = None
        if self.mamba_interaction == 'interleaved_fused':
            first_stage = self._mamba_layer_range(
                mamba_events, 0, 1, finalize=False
            )
            short_memory, memory_gate = self._short_memory(
                events, first_stage
            )
            if memory_gate is not None:
                preliminary_gates['short_memory_state'] = (
                    self._gather_last(memory_gate, batch['seq_len'])
                )
            precomputed_short = self.short_term(
                short_memory,
                batch['seq_len'],
                short_target,
                history_keys=short_history_keys,
                target_keys=short_target_keys,
            )
            if self.target_history is not None:
                precomputed_short, target_history_gates = (
                    self.target_history.forward_target(
                        short_memory,
                        batch['concept_seq'],
                        batch['seq_len'],
                        target_ids,
                        short_target,
                        precomputed_short,
                    )
                )
                preliminary_gates.update(target_history_gates)
            last_stage = self._gather_last(first_stage, batch['seq_len'])
            conditioned_last, conditioning_gate = (
                self.mamba_input_conditioner(
                    last_stage, precomputed_short, target
                )
            )
            positions = torch.arange(
                first_stage.size(1), device=events.device
            ).unsqueeze(0)
            last_positions = (batch['seq_len'] - 1).clamp_min(0).unsqueeze(1)
            second_stage_input = torch.where(
                (positions == last_positions).unsqueeze(-1),
                conditioned_last.unsqueeze(1),
                first_stage,
            )
            interleaved_encoded = self._mamba_layer_range(
                second_stage_input,
                1,
                len(self.sequence_encoder.layers),
                finalize=True,
            )
            preliminary_gates['mamba_local_conditioning'] = conditioning_gate
            events_for_mamba = events
        elif self.mamba_input_conditioner is not None:
            precomputed_short = self.short_term(
                events,
                batch['seq_len'],
                short_target,
                history_keys=short_history_keys,
                target_keys=short_target_keys,
            )
            if self.target_history is not None:
                precomputed_short, target_history_gates = (
                    self.target_history.forward_target(
                        events,
                        batch['concept_seq'],
                        batch['seq_len'],
                        target_ids,
                        short_target,
                        precomputed_short,
                    )
                )
                preliminary_gates.update(target_history_gates)
            last_event = self._gather_last(mamba_events, batch['seq_len'])
            conditioned_last, conditioning_gate = (
                self.mamba_input_conditioner(
                    last_event, precomputed_short, target
                )
            )
            positions = torch.arange(
                events.size(1), device=events.device
            ).unsqueeze(0)
            last_positions = (batch['seq_len'] - 1).clamp_min(0).unsqueeze(1)
            events_for_mamba = torch.where(
                (positions == last_positions).unsqueeze(-1),
                conditioned_last.unsqueeze(1),
                events,
            )
            preliminary_gates['mamba_local_conditioning'] = conditioning_gate
        else:
            events_for_mamba = mamba_events
        if self.dynamics_mode == 'factorized':
            exposure, acquisition, encoded = self._factorized_states(
                batch, concept_states, events
            )
            sequence = self.factorized_fusion(
                self._gather_last(exposure, batch['seq_len']),
                self._gather_last(acquisition, batch['seq_len']),
                target,
            )
        else:
            # See forward_sequence: this path is intentionally Mamba-free for
            # the hard Transformer-only ablation.
            encoded = events
            long_state = None
            if self.has_mamba:
                encoded = (
                    interleaved_encoded
                    if interleaved_encoded is not None
                    else self.sequence_encoder(events_for_mamba)
                )
                long_state = self._condition_sequence(
                    self._gather_last(encoded, batch['seq_len']), target
                )
                if routed_student is not None:
                    long_state, preliminary_gates['student_long'] = (
                        self.student_long_adapter(
                            long_state, target, *routed_student
                        )
                    )
            if self.short_term is not None:
                if precomputed_short is None:
                    if self.has_mamba:
                        short_memory, memory_gate = self._short_memory(
                            events, encoded
                        )
                    else:
                        short_memory, memory_gate = events, None
                    if memory_gate is not None:
                        preliminary_gates['short_memory_state'] = (
                            self._gather_last(memory_gate, batch['seq_len'])
                        )
                    short_state = self.short_term(
                        short_memory,
                        batch['seq_len'],
                        short_target,
                        history_keys=short_history_keys,
                        target_keys=short_target_keys,
                    )
                    if self.target_history is not None:
                        short_state, target_history_gates = (
                            self.target_history.forward_target(
                                short_memory,
                                batch['concept_seq'],
                                batch['seq_len'],
                                target_ids,
                                short_target,
                                short_state,
                            )
                        )
                        preliminary_gates.update(target_history_gates)
                else:
                    short_state = precomputed_short
                if routed_transition is not None:
                    short_state, preliminary_gates['transition_short'] = (
                        self.transition_short_adapter(
                            short_state, target, *routed_transition
                        )
                    )
                if not self.has_mamba:
                    sequence = short_state
                elif self.mamba_interaction == 'local_to_mamba':
                    sequence = long_state
                else:
                    sequence, branch_gates = self.branch_fusion(
                        long_state, short_state, target
                    )
                    preliminary_gates.update(branch_gates)
            else:
                sequence = long_state
        if self.dynamics_mode == 'factorized':
            sequence = self._condition_sequence(sequence, target)
        evidence = {}
        reported_evidence = {}

        if self.student_graph_encoder is not None:
            student_evidence = routed_student
            if student_evidence is None:
                student_evidence = (
                    student_profile,
                    student_profile.new_ones(student_profile.size(0), 1),
                )
            reported_evidence['student_graph'] = student_evidence
            if routed_student is None or self.keep_routed_posthoc:
                evidence['student_graph'] = student_evidence

        if self.ability_mode in {'evidence', 'both'}:
            evidence['ability'] = ability['evidence']

        if self.use_target_retrieval:
            memory = self._retrieval_memory(events, encoded)
            memory_ids = batch['concept_seq']
            if self.retrieval_scope == 'adaptive':
                broad = self.target_retrieval.forward_target(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope='all',
                )
                local = self.target_retrieval.forward_target(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope='same',
                )
                reported_evidence['broad_retrieval'] = broad
                reported_evidence['local_retrieval'] = local
                sequence, weights = self.history_router(
                    sequence, target, broad[0], local[0], local[1]
                )
                preliminary_gates.update({
                    'history_global': weights[..., 0:1],
                    'history_broad': weights[..., 1:2],
                    'history_local': weights[..., 2:3],
                })
            else:
                evidence['retrieval'] = self.target_retrieval.forward_target(
                    target,
                    memory,
                    batch['seq_len'],
                    target_ids=target_ids,
                    memory_ids=memory_ids,
                    scope=self.retrieval_scope,
                )

        if self.use_concept_memory:
            state, reliability = self.concept_memory.forward_target(
                batch, concept_states, success, events
            )
            reported_evidence['concept_memory'] = (state, reliability)
            if self.concept_memory_fusion == 'competitive':
                sequence, preliminary_gates['concept_memory'] = (
                    self.coverage_fusion(
                        sequence, target, state, reliability
                    )
                )
            else:
                evidence['concept_memory'] = (state, reliability)

        if self.use_item_context and self.item_evidence.available:
            target_question = batch.get('target_question')
            if target_question is None:
                raise ValueError('item context requires target_question')
            item_evidence = self.item_evidence(
                target_question,
                target_ids,
                concept_states,
                success,
            )
            reported_evidence['item_hierarchy'] = item_evidence
            if torch.any(item_evidence[1] > 0):
                evidence['item_hierarchy'] = item_evidence

        if (
            self.use_population_graph
            and self.population_fusion in {'evidence', 'both'}
        ):
            states, reliabilities = self.population_evidence.forward_sequence(
                batch, population_states, success
            )
            evidence['population'] = (
                self._gather_last(states, batch['seq_len']),
                self._gather_last(reliabilities, batch['seq_len']),
            )
        if self.use_mastery:
            attempts, correct = self._target_counts(batch)
            evidence['mastery'] = self.mastery_evidence(
                target_ids, attempts, correct, success
            )
        if self.use_transition_graph:
            transition_evidence = routed_transition
            if transition_evidence is None:
                outcome = self._outcome_states(
                    batch['concept_seq'],
                    batch['response_seq'],
                    success,
                    attempts=event_attempts,
                    correct=event_correct,
                )
                transition_evidence = (
                    self.transition_evidence.forward_target(
                        batch['concept_seq'],
                        target_ids,
                        batch['seq_len'],
                        concept_states,
                        outcome,
                        edge_valid_mask=self._target_transition_mask(batch),
                    )
                )
            reported_evidence['transition'] = transition_evidence
            if routed_transition is None or self.keep_routed_posthoc:
                evidence['transition'] = transition_evidence

        fused, gates = self.evidence_fusion(sequence, target, evidence)
        gates = {**preliminary_gates, **gates}
        reported_evidence.update(evidence)
        logits = self.predictor(torch.cat([
            fused, target, fused * target
        ], dim=-1)).squeeze(-1)
        return self._format_output(
            logits, fused, reported_evidence, gates, return_aux
        )


# Compatibility name for development checkpoints created before the modular
# interface was introduced. Official V8 configs use ``modular_evidence_kt``.
InductiveCausalDualGraphKT = ModularEvidenceKT
