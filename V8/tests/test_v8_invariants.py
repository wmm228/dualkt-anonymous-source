import copy
import unittest

import torch

from V8.model import (
    BaselineBranchFusion,
    CausalTargetTransitionGraph,
    DGMKTStudentHypergraphProfile,
    DualBranchStateFusion,
    HierarchicalItemEvidence,
    ModularEvidenceKT,
    TargetAlignedHistoryAttention,
)


def make_model(
    difficulty=True,
    population=True,
    mastery=True,
    transition=True,
    target_readout=True,
    concept_memory=False,
    memory_fusion='competitive',
    population_fusion='evidence',
    retrieval=False,
    retrieval_source='event',
    retrieval_scope='all',
    dynamics_mode='entangled',
    ability_mode='off',
    temporal_backbone='mamba',
    branch_fusion='orthogonal_innovation',
    short_memory_mode='contiguous',
    target_history_size=3,
    student_graph=False,
    student_incidence=None,
    statistics=None,
    evidence_placement='posthoc',
    outcome_calibration='static',
    transition_aggregation='uniform',
    transition_decay=0.97,
    outcome_prior_strength=None,
    short_memory_source='event',
    mamba_interaction='parallel',
    mamba_layers=1,
    item_context=False,
    item_statistics=None,
    item_incidence=None,
    attempt_grouped_transition=False,
    student_conditioning='off',
):
    graph = torch.eye(9)
    graph[2, 3] = graph[3, 2] = 0.25
    if statistics is None:
        statistics = torch.zeros(9, 2)
        statistics[:, 0] = 0.5
        statistics[:, 1] = 0.8
        statistics[0, 1] = 0.0
    if student_graph and student_incidence is None:
        student_incidence = torch.zeros(6, 9)
        student_incidence[1, 2] = 3.0
        student_incidence[1, 3] = 1.0
        student_incidence[2, 2] = 1.0
        student_incidence[2, 4] = 4.0
    if item_context and item_statistics is None:
        item_statistics = torch.zeros(7, 3)
        item_statistics[:, 0] = 0.5
        item_statistics[2:, 1] = 0.8
        item_statistics[2:, 2] = 0.5
        item_incidence = torch.zeros(7, 9)
        item_incidence[2:, 2] = 1.0
    return ModularEvidenceKT(
        n_questions=7 if item_context else 0,
        n_concepts=9,
        d_model=16,
        d_state=4,
        d_conv=2,
        expand=1,
        dropout=0.0,
        mamba_layers=mamba_layers,
        temporal_backbone=temporal_backbone,
        n_heads=4,
        n_layers=2,
        short_window=3,
        summary_block_size=2,
        max_seq_len=8,
        short_memory_mode=short_memory_mode,
        target_history_size=target_history_size,
        branch_fusion=branch_fusion,
        short_memory_source=short_memory_source,
        mamba_interaction=mamba_interaction,
        dynamics_mode=dynamics_mode,
        concept_graph=graph,
        concept_statistics=statistics,
        use_difficulty=difficulty,
        use_student_graph=student_graph,
        student_incidence=student_incidence,
        use_population_graph=population,
        population_fusion=population_fusion,
        use_mastery=mastery,
        use_concept_memory=concept_memory,
        use_transition_graph=transition,
        transition_aggregation=transition_aggregation,
        transition_decay=transition_decay,
        outcome_calibration=outcome_calibration,
        outcome_prior_strength=outcome_prior_strength,
        evidence_placement=evidence_placement,
        use_target_retrieval=retrieval,
        retrieval_source=retrieval_source,
        retrieval_scope=retrieval_scope,
        ability_mode=ability_mode,
        target_conditioned_readout=target_readout,
        concept_memory_fusion=memory_fusion,
        use_item_context=item_context,
        item_statistics=item_statistics,
        item_concept_incidence=item_incidence,
        attempt_grouped_transition=attempt_grouped_transition,
        student_conditioning=student_conditioning,
    )


def make_batch(length=6, padded_to=None):
    padded_to = padded_to or length
    concepts = torch.zeros(1, padded_to, dtype=torch.long)
    responses = torch.zeros(1, padded_to)
    concepts[0, :length] = torch.tensor([2, 3, 2, 4, 2, 3][:length])
    responses[0, :length] = torch.tensor([1., 0., 1., 0., 0., 1.][:length])
    return {
        'concept_seq': concepts,
        'response_seq': responses,
        'seq_len': torch.tensor([length]),
        'predict_mask': (
            torch.arange(max(padded_to - 1, 0)).unsqueeze(0) < length - 1
        ),
        'initial_concept_attempts': torch.zeros(1, 9),
        'initial_concept_correct': torch.zeros(1, 9),
    }


class V8InvariantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = make_model().eval()

    def test_current_target_response_is_not_an_input(self):
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = self.model.forward_sequence(batch)
            mutated = self.model.forward_sequence(changed)
        torch.testing.assert_close(original, mutated)

    def test_hierarchical_item_evidence_is_reliable_and_trainable(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            item_context=True,
        ).train()
        batch = make_batch()
        batch['question_seq'] = torch.tensor([[2, 3, 2, 4, 2, 3]])
        output = model.forward_sequence(batch, return_aux=True)
        self.assertIn('item_hierarchy', output['evidence_states'])
        torch.testing.assert_close(
            output['evidence_reliability']['item_hierarchy'],
            torch.full((1, 5, 1), 0.8),
        )
        output['logits'].sum().backward()
        gradient = model.item_evidence.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_zero_reliability_item_data_exactly_falls_back(self):
        statistics = torch.zeros(7, 3)
        statistics[:, 0] = 0.5
        incidence = torch.zeros(7, 9)
        torch.manual_seed(97)
        reference = make_model(
            population=False, mastery=False, transition=False
        ).eval()
        torch.manual_seed(97)
        augmented = make_model(
            population=False,
            mastery=False,
            transition=False,
            item_context=True,
            item_statistics=statistics,
            item_incidence=incidence,
        ).eval()
        self.assertIsNone(augmented.item_evidence)
        batch = make_batch()
        batch['question_seq'] = torch.zeros(1, 6, dtype=torch.long)
        with torch.no_grad():
            expected = reference.forward_sequence(batch)
            actual = augmented.forward_sequence(batch)
        torch.testing.assert_close(expected, actual)

    def test_attempt_boundary_filters_within_item_transition(self):
        module = CausalTargetTransitionGraph(
            d_model=4, reliability_strength=5.0, dropout=0.0
        ).eval()
        concept_states = torch.randn(6, 4)
        concept_seq = torch.tensor([[2, 3, 4]])
        outcome = torch.randn(1, 2, 4)
        with torch.no_grad():
            _, reliability = module.forward_sequence(
                concept_seq,
                torch.tensor([3]),
                concept_states,
                outcome,
                edge_valid_mask=torch.tensor([[False, True]]),
            )
        torch.testing.assert_close(reliability[0, 0], torch.tensor([0.0]))
        torch.testing.assert_close(
            reliability[0, 1], torch.tensor([1.0 / 6.0])
        )

    def test_future_event_does_not_change_earlier_predictions(self):
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 6
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = self.model.forward_sequence(batch)
            mutated = self.model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

    def test_multi_concept_question_uses_padding_safe_mean_pooling(self):
        model = make_model(
            difficulty=False,
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            mamba_layers=2,
        ).eval()
        batch = make_batch()
        bundled = copy.deepcopy(batch)
        bundled['concept_seq'] = torch.stack([
            batch['concept_seq'], torch.zeros_like(batch['concept_seq'])
        ], dim=-1)
        with torch.no_grad():
            scalar = model.forward_sequence(batch)
            pooled = model.forward_sequence(bundled)
        torch.testing.assert_close(scalar, pooled)

    def test_student_identifier_and_legacy_side_features_are_not_inputs(self):
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['student_id'] = torch.tensor([987654321])
        changed['difficulty_seq'] = torch.randn(1, 6, 4) * 100.0
        changed['student_profile_seq'] = torch.randn(1, 6, 20) * 100.0
        with torch.no_grad():
            original = self.model.forward_sequence(batch)
            mutated = self.model.forward_sequence(changed)
        torch.testing.assert_close(original, mutated)

    def test_dual_branch_ignores_student_identifier_and_legacy_profiles(self):
        model = make_model(
            temporal_backbone='mamba_transformer'
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['student_id'] = torch.tensor([987654321])
        changed['student_profile_seq'] = torch.randn(1, 6, 20) * 100.0
        changed['target_profile'] = torch.randn(1, 20) * 100.0
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original, mutated)

    def test_dgmkt_student_profile_matches_released_hgnn_formula(self):
        incidence = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
        ])
        module = DGMKTStudentHypergraphProfile(incidence, d_model=4)
        student_degree = incidence.sum(dim=1).clamp_min(1.0)
        concept_degree = incidence.sum(dim=0).clamp_min(1.0)
        scale = student_degree.rsqrt()
        graph = (
            scale.unsqueeze(1)
            * incidence
            / concept_degree.unsqueeze(0)
        ).matmul(incidence.T * scale.unsqueeze(0))
        transformed = module.student_anchor.matmul(module.weight) + module.bias
        expected = torch.relu(graph.matmul(transformed))
        torch.testing.assert_close(module.table(), expected)

    def test_dgmkt_student_id_profile_is_indexed_and_trainable(self):
        model = make_model(
            difficulty=False,
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_mode='contiguous_target',
            student_graph=True,
        ).eval()
        first = make_batch()
        second = copy.deepcopy(first)
        first['student_id'] = torch.tensor([2])
        second['student_id'] = torch.tensor([3])
        with torch.no_grad():
            first_output = model.forward_sequence(first, return_aux=True)
            second_output = model.forward_sequence(second, return_aux=True)
        self.assertFalse(torch.allclose(
            first_output['logits'], second_output['logits']
        ))
        self.assertIn('student_graph', first_output['evidence_states'])

        model.train()
        output = model.forward_sequence(first, return_aux=True)
        output['logits'].sum().backward()
        gradient = model.student_graph_encoder.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_student_graph_does_not_shift_shared_initialization(self):
        torch.manual_seed(79)
        reference = make_model(
            difficulty=False,
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_mode='contiguous_target',
            student_graph=False,
        )
        torch.manual_seed(79)
        augmented = make_model(
            difficulty=False,
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_mode='contiguous_target',
            student_graph=True,
        )
        augmented_state = augmented.state_dict()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, augmented_state[name])

    def test_dual_branch_is_causal_and_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
        ).train()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        model.eval()
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original, mutated)

        model.train()
        output = model.forward_sequence(batch, return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.short_term.layers[0].query.weight.grad,
            model.branch_fusion.orthogonal_gates.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(
            {
                'branch_short_weight',
                'branch_shared',
                'branch_innovation',
                'branch_novelty',
            },
            set(output['evidence_gates']),
        )

    def test_local_to_mamba_is_causal_and_receives_task_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_mode='contiguous_target',
            short_memory_source='event',
            mamba_interaction='local_to_mamba',
        )
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        model.eval()
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

        model.train()
        output = model.forward_sequence(batch, return_aux=True)
        self.assertIn(
            'mamba_local_conditioning', output['evidence_gates']
        )
        self.assertNotIn('branch_short_weight', output['evidence_gates'])
        output['logits'].sum().backward()
        for gradient in (
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.short_term.layers[0].query.weight.grad,
            model.mamba_input_conditioner.update[0].weight.grad,
            model.mamba_input_conditioner.gate[-1].weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_local_to_mamba_fused_keeps_the_late_fusion_path(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_source='event',
            mamba_interaction='local_to_mamba_fused',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        self.assertIn(
            'mamba_local_conditioning', output['evidence_gates']
        )
        self.assertIn('branch_short_weight', output['evidence_gates'])
        output['logits'].sum().backward()
        gradient = model.branch_fusion.vector_router.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_local_to_mamba_supports_single_target_inference(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            short_memory_source='event',
            mamba_interaction='local_to_mamba',
        ).eval()
        batch = make_batch()
        batch['target_concept'] = torch.tensor([4])
        with torch.no_grad():
            output = model(batch, return_aux=True)
        self.assertEqual(tuple(output['logits'].shape), (1,))
        self.assertIn(
            'mamba_local_conditioning', output['evidence_gates']
        )

    def test_local_to_mamba_rejects_circular_state_memory(self):
        with self.assertRaisesRegex(ValueError, 'requires event short memory'):
            make_model(
                temporal_backbone='mamba_transformer',
                short_memory_source='mean',
                mamba_interaction='local_to_mamba',
            )

    def test_local_to_mamba_preserves_shared_initialization(self):
        kwargs = {
            'population': False,
            'mastery': False,
            'transition': False,
            'temporal_backbone': 'mamba_transformer',
            'short_memory_source': 'event',
        }
        torch.manual_seed(131)
        reference = make_model(**kwargs)
        torch.manual_seed(131)
        conditioned = make_model(
            **kwargs, mamba_interaction='local_to_mamba'
        )
        conditioned_state = conditioned.state_dict()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, conditioned_state[name])

    def test_interleaved_mamba_is_causal_and_trains_both_stages(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_mode='contiguous_target',
            short_memory_source='mean',
            mamba_interaction='interleaved_fused',
            mamba_layers=2,
        )
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        model.eval()
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

        model.train()
        output = model.forward_sequence(batch, return_aux=True)
        self.assertIn('short_memory_state', output['evidence_gates'])
        self.assertIn('mamba_local_conditioning', output['evidence_gates'])
        self.assertIn('branch_short_weight', output['evidence_gates'])
        output['logits'].sum().backward()
        for gradient in (
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.sequence_encoder.layers[1].cpu_fallback.weight_ih_l0.grad,
            model.short_term.layers[0].query.weight.grad,
            model.mamba_input_conditioner.update[0].weight.grad,
            model.branch_fusion.vector_router.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_interleaved_mamba_supports_all_controlled_memory_sources(self):
        batch = make_batch()
        batch['target_concept'] = torch.tensor([4])
        for source in ('event', 'mean', 'state'):
            model = make_model(
                population=False,
                mastery=False,
                transition=False,
                temporal_backbone='mamba_transformer',
                short_memory_source=source,
                mamba_interaction='interleaved_fused',
                mamba_layers=2,
            ).eval()
            with torch.no_grad():
                sequence = model.forward_sequence(batch, return_aux=True)
                target = model(batch, return_aux=True)
            self.assertEqual(
                tuple(sequence['logits'].shape),
                (1, batch['concept_seq'].size(1) - 1),
            )
            self.assertEqual(tuple(target['logits'].shape), (1,))
            self.assertIn(
                'mamba_local_conditioning', sequence['evidence_gates']
            )

    def test_interleaved_mamba_requires_two_layers(self):
        with self.assertRaisesRegex(ValueError, 'at least two Mamba layers'):
            make_model(
                temporal_backbone='mamba_transformer',
                mamba_interaction='interleaved_fused',
                mamba_layers=1,
            )

    def test_student_profile_conditioning_requires_student_graph(self):
        with self.assertRaisesRegex(ValueError, 'requires student graph'):
            make_model(
                temporal_backbone='mamba_transformer',
                student_conditioning='mamba',
            )

    def test_student_profile_conditioners_are_causal_and_trainable(self):
        for mode, expected_gates in (
            ('mamba', {'student_mamba_conditioning'}),
            ('transformer', {'student_transformer_conditioning'}),
            ('both', {
                'student_mamba_conditioning',
                'student_transformer_conditioning',
            }),
        ):
            model = make_model(
                population=False,
                mastery=False,
                transition=False,
                temporal_backbone='mamba_transformer',
                branch_fusion='adaptive_orthogonal_vector',
                short_memory_mode='contiguous_target',
                mamba_interaction='local_to_mamba_fused',
                mamba_layers=2,
                student_graph=True,
                student_conditioning=mode,
            )
            batch = make_batch()
            batch['student_id'] = torch.tensor([2])
            changed = copy.deepcopy(batch)
            changed['concept_seq'][0, -1] = 7
            changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
            model.eval()
            with torch.no_grad():
                original = model.forward_sequence(batch)
                mutated = model.forward_sequence(changed)
            torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

            model.train()
            output = model.forward_sequence(batch, return_aux=True)
            self.assertTrue(expected_gates.issubset(output['evidence_gates']))
            output['logits'].sum().backward()
            conditioners = [
                module for module in (
                    model.student_mamba_conditioner,
                    model.student_transformer_conditioner,
                ) if module is not None
            ]
            for conditioner in conditioners:
                gradient = conditioner.update[-1].weight.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_student_profile_conditioning_supports_single_target(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            mamba_interaction='local_to_mamba_fused',
            mamba_layers=2,
            student_graph=True,
            student_conditioning='both',
        ).eval()
        batch = make_batch()
        batch['student_id'] = torch.tensor([2])
        batch['target_concept'] = torch.tensor([4])
        with torch.no_grad():
            output = model(batch, return_aux=True)
        self.assertEqual(tuple(output['logits'].shape), (1,))
        self.assertIn(
            'student_mamba_conditioning', output['evidence_gates']
        )
        self.assertIn(
            'student_transformer_conditioning', output['evidence_gates']
        )

    def test_student_profile_conditioning_preserves_control_initialization(self):
        kwargs = {
            'population': False,
            'mastery': False,
            'transition': False,
            'temporal_backbone': 'mamba_transformer',
            'mamba_interaction': 'local_to_mamba_fused',
            'mamba_layers': 2,
            'student_graph': True,
        }
        torch.manual_seed(149)
        reference = make_model(**kwargs).eval()
        torch.manual_seed(149)
        conditioned = make_model(
            **kwargs, student_conditioning='both'
        ).eval()
        conditioned_state = conditioned.state_dict()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, conditioned_state[name])

        batch = make_batch()
        batch['student_id'] = torch.tensor([2])
        with torch.no_grad():
            expected = reference.forward_sequence(batch)
            actual = conditioned.forward_sequence(batch)
        torch.testing.assert_close(expected, actual)

    def test_adaptive_routers_start_from_the_mean_fusion(self):
        fusion = DualBranchStateFusion(
            d_model=16,
            mode='mean',
            dropout=0.0,
        ).eval()
        long_state = torch.randn(2, 3, 16)
        short_state = torch.randn(2, 3, 16)
        target = torch.randn(2, 3, 16)
        with torch.no_grad():
            expected, _ = fusion(long_state, short_state, target)
            for mode in (
                'adaptive_scalar',
                'adaptive_vector',
                'target_competitive',
            ):
                fusion.mode = mode
                actual, diagnostics = fusion(
                    long_state, short_state, target
                )
                torch.testing.assert_close(expected, actual)
                torch.testing.assert_close(
                    diagnostics['branch_short_weight'],
                    torch.full_like(
                        diagnostics['branch_short_weight'], 0.5
                    ),
                )

    def test_mean_fusion_bypasses_the_complete_orthogonal_module(self):
        fusion = DualBranchStateFusion(
            d_model=8,
            mode='mean',
            dropout=0.0,
        ).train()
        fusion._orthogonal_components = lambda *_: self.fail(
            'fixed mean must not execute orthogonal decomposition'
        )
        long_state = torch.randn(2, 3, 8, requires_grad=True)
        short_state = torch.randn(2, 3, 8, requires_grad=True)
        target = torch.randn(2, 3, 8)
        output, diagnostics = fusion(long_state, short_state, target)
        expected = fusion.output_norm(0.5 * (
            fusion.long_norm(long_state) + fusion.short_norm(short_state)
        ))
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertIsNone(fusion.vector_router.weight.grad)
        self.assertIsNone(fusion.adaptive_update_gates.weight.grad)
        self.assertIsNone(fusion.shared_update[0].weight.grad)
        self.assertIsNone(fusion.novel_update[0].weight.grad)
        torch.testing.assert_close(
            diagnostics['branch_short_weight'],
            torch.full_like(diagnostics['branch_short_weight'], 0.5),
        )
        torch.testing.assert_close(
            diagnostics['branch_innovation'],
            torch.zeros_like(diagnostics['branch_innovation']),
        )

    def test_physical_mean_ablation_contains_no_orthogonal_fusion(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='fixed_mean_no_orthogonal_fusion',
        ).train()
        fusion = model.branch_fusion
        self.assertIsInstance(fusion, BaselineBranchFusion)
        for removed in (
            'gate_context',
            'vector_router',
            'adaptive_update_gates',
            'shared_update',
            'novel_update',
            'innovation_relevance',
        ):
            self.assertFalse(hasattr(fusion, removed))
        long_state = torch.randn(2, 3, 16)
        short_state = torch.randn(2, 3, 16)
        target = torch.randn(2, 3, 16)
        actual, _ = fusion(long_state, short_state, target)
        expected = fusion.output_norm(0.5 * (
            fusion.long_norm(long_state) + fusion.short_norm(short_state)
        ))
        torch.testing.assert_close(actual, expected)

    def test_physical_concat_ablation_learns_only_linear_fusion(self):
        fusion = BaselineBranchFusion(
            d_model=16,
            mode='concat_linear_no_orthogonal_fusion',
        ).train()
        long_state = torch.randn(2, 3, 16, requires_grad=True)
        short_state = torch.randn(2, 3, 16, requires_grad=True)
        output, diagnostics = fusion(
            long_state, short_state, torch.randn(2, 3, 16)
        )
        output.square().sum().backward()
        self.assertIsNotNone(fusion.concat_projection.weight.grad)
        self.assertGreater(
            float(fusion.concat_projection.weight.grad.abs().sum()), 0.0
        )
        torch.testing.assert_close(
            diagnostics['branch_innovation'],
            torch.zeros_like(diagnostics['branch_innovation']),
        )

    def test_normalized_weighted_sum_starts_at_mean_and_learns(self):
        fusion = BaselineBranchFusion(
            d_model=16,
            mode='normalized_channel_weighted_sum',
        ).train()
        long_state = torch.randn(2, 3, 16, requires_grad=True)
        short_state = torch.randn(2, 3, 16, requires_grad=True)
        output, diagnostics = fusion(
            long_state, short_state, torch.randn(2, 3, 16)
        )
        expected = fusion.output_norm(0.5 * (
            fusion.long_norm(long_state) + fusion.short_norm(short_state)
        ))
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        self.assertIsNotNone(fusion.branch_logits.grad)
        self.assertIsNotNone(fusion.weighted_projection.weight.grad)
        torch.testing.assert_close(
            diagnostics['branch_short_weight'],
            torch.full_like(diagnostics['branch_short_weight'], 0.5),
        )

    def test_adaptive_orthogonal_router_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.short_term.layers[0].query.weight.grad,
            model.branch_fusion.scalar_router.weight.grad,
            model.branch_fusion.adaptive_update_gates.weight.grad,
            model.branch_fusion.shared_update[0].weight.grad,
            model.branch_fusion.novel_update[0].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_legacy_fusion_checkpoint_remains_strictly_loadable(self):
        old = DualBranchStateFusion(
            d_model=16,
            mode='mean',
            dropout=0.0,
        )
        state = {
            key: value for key, value in old.state_dict().items()
            if not key.startswith((
                'scalar_router.',
                'vector_router.',
                'competitive_score.',
                'adaptive_update_gates.',
                'innovation_relevance.',
            ))
        }
        restored = DualBranchStateFusion(
            d_model=16,
            mode='mean',
            dropout=0.0,
        )
        restored.load_state_dict(state, strict=True)
        adaptive = DualBranchStateFusion(
            d_model=16,
            mode='adaptive_vector',
            dropout=0.0,
        )
        with self.assertRaises(RuntimeError):
            adaptive.load_state_dict(state, strict=True)

    def test_denoised_orthogonal_fusion_has_one_gated_innovation_path(self):
        fusion = DualBranchStateFusion(
            d_model=4,
            mode='denoised_adaptive_orthogonal_vector',
            dropout=0.0,
        ).eval()
        long_state = torch.randn(2, 3, 4)
        short_state = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        with torch.no_grad():
            output, diagnostics = fusion(
                long_state, short_state, target
            )
        self.assertEqual(output.shape, long_state.shape)
        self.assertTrue(torch.isfinite(output).all())
        relevance = diagnostics['branch_innovation_relevance']
        torch.testing.assert_close(relevance, torch.ones_like(relevance))

        fusion.output_norm = torch.nn.Identity()
        with torch.no_grad():
            fusion.innovation_relevance.bias.fill_(-100.0)
            fusion.adaptive_update_gates.bias.fill_(-100.0)
        fixed_long = torch.randn(2, 3, 4)
        fixed_overlap = torch.randn(2, 3, 4)
        fusion._orthogonal_components = lambda long, short: (
            fixed_long,
            short,
            fixed_overlap,
            short,
            torch.ones_like(short[..., :1]),
        )
        first, _ = fusion(
            fixed_long, torch.randn(2, 3, 4), target
        )
        second, _ = fusion(
            fixed_long, torch.randn(2, 3, 4), target
        )
        torch.testing.assert_close(first, second)

    def test_denoised_fusion_is_rng_neutral_nested_incumbent(self):
        torch.manual_seed(79)
        incumbent = DualBranchStateFusion(
            d_model=8,
            mode='adaptive_orthogonal_vector',
            dropout=0.0,
        ).eval()
        torch.manual_seed(79)
        candidate = DualBranchStateFusion(
            d_model=8,
            mode='denoised_adaptive_orthogonal_vector',
            dropout=0.0,
        ).eval()
        candidate_state = candidate.state_dict()
        for name, value in incumbent.state_dict().items():
            torch.testing.assert_close(value, candidate_state[name])

        long_state = torch.randn(2, 3, 8)
        short_state = torch.randn(2, 3, 8)
        target = torch.randn(2, 3, 8)
        with torch.no_grad():
            incumbent_output, _ = incumbent(long_state, short_state, target)
            candidate_output, diagnostics = candidate(
                long_state, short_state, target
            )
        torch.testing.assert_close(incumbent_output, candidate_output)
        torch.testing.assert_close(
            diagnostics['branch_innovation_relevance'],
            torch.ones_like(
                diagnostics['branch_innovation_relevance']
            ),
        )

    def test_denoised_orthogonal_fusion_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='denoised_adaptive_orthogonal_vector',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.branch_fusion.vector_router.weight.grad,
            model.branch_fusion.adaptive_update_gates.weight.grad,
            model.branch_fusion.shared_update[0].weight.grad,
            model.branch_fusion.novel_update[0].weight.grad,
            model.branch_fusion.innovation_relevance.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_raw_short_ablation_changes_only_the_orthogonal_decomposition(self):
        torch.manual_seed(197)
        full = DualBranchStateFusion(
            d_model=8,
            mode='denoised_adaptive_orthogonal_vector',
            dropout=0.0,
        ).eval()
        torch.manual_seed(197)
        ablated = DualBranchStateFusion(
            d_model=8,
            mode='denoised_adaptive_raw_short_vector',
            dropout=0.0,
        ).eval()
        self.assertEqual(full.state_dict().keys(), ablated.state_dict().keys())
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, ablated.state_dict()[name])
        self.assertEqual(
            sum(parameter.numel() for parameter in full.parameters()),
            sum(parameter.numel() for parameter in ablated.parameters()),
        )

        long_state = torch.randn(2, 3, 8)
        short_state = torch.randn(2, 3, 8)
        target = torch.randn(2, 3, 8)
        with torch.no_grad():
            full_output, full_diagnostics = full(
                long_state, short_state, target
            )
            ablated_output, ablated_diagnostics = ablated(
                long_state, short_state, target
            )
        self.assertFalse(torch.equal(full_output, ablated_output))
        torch.testing.assert_close(
            ablated_diagnostics['branch_novelty'],
            torch.ones_like(ablated_diagnostics['branch_novelty']),
        )
        torch.testing.assert_close(
            full_diagnostics['branch_innovation_relevance'],
            ablated_diagnostics['branch_innovation_relevance'],
        )

    def test_raw_short_ablation_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='denoised_adaptive_raw_short_vector',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.branch_fusion.vector_router.weight.grad,
            model.branch_fusion.adaptive_update_gates.weight.grad,
            model.branch_fusion.shared_update[0].weight.grad,
            model.branch_fusion.novel_update[0].weight.grad,
            model.branch_fusion.innovation_relevance.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_target_history_exactly_falls_back_without_matches(self):
        module = TargetAlignedHistoryAttention(
            d_model=16,
            n_heads=4,
            max_matches=3,
            dropout=0.0,
        ).eval()
        history = torch.randn(1, 4, 16)
        concepts = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[8, 8, 8]])
        target_state = torch.randn(1, 3, 16)
        recent = torch.randn(1, 3, 16)
        with torch.no_grad():
            output, diagnostics = module.forward_sequence(
                history,
                concepts,
                torch.tensor([4]),
                targets,
                target_state,
                recent,
            )
        torch.testing.assert_close(output, recent)
        torch.testing.assert_close(
            diagnostics['target_history_coverage'],
            torch.zeros_like(diagnostics['target_history_coverage']),
        )

    def test_target_history_does_not_shift_shared_parameter_initialization(self):
        torch.manual_seed(73)
        contiguous = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_vector',
            short_memory_mode='contiguous',
        )
        torch.manual_seed(73)
        augmented = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_vector',
            short_memory_mode='contiguous_target',
        )
        augmented_state = augmented.state_dict()
        for name, value in contiguous.state_dict().items():
            torch.testing.assert_close(value, augmented_state[name])

    def test_target_history_is_causal_padding_safe_and_trainable(self):
        torch.manual_seed(31)
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_vector',
            short_memory_mode='contiguous_target',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 6
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
            short = model.forward_sequence(make_batch(length=4, padded_to=4))
            padded = model.forward_sequence(
                make_batch(length=4, padded_to=7)
            )[:, :3]
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])
        torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-6)

        model.train()
        output = model.forward_sequence(batch, return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.target_history.layer.query.weight.grad,
            model.target_history.update[0].weight.grad,
            model.target_history.gate[-1].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertIn('target_history', output['evidence_gates'])
        self.assertIn('target_history_coverage', output['evidence_gates'])

    def test_dual_branch_future_event_does_not_change_earlier_predictions(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 6
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

    def test_dual_branch_right_padding_does_not_change_valid_predictions(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            temporal_backbone='mamba_transformer',
        ).eval()
        short = make_batch(length=4, padded_to=4)
        padded = make_batch(length=4, padded_to=7)
        with torch.no_grad():
            expected = model.forward_sequence(short)
            actual = model.forward_sequence(padded)[:, :3]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_short_innovation_is_orthogonal_to_the_long_direction(self):
        fusion = DualBranchStateFusion(
            d_model=16,
            mode='orthogonal_innovation',
            dropout=0.0,
        ).eval()
        long_state = torch.randn(2, 3, 16)
        short_state = torch.randn(2, 3, 16)
        normalized_long, _, _, novel, _ = fusion._orthogonal_components(
            long_state, short_state
        )
        direction = torch.nn.functional.normalize(
            normalized_long, dim=-1, eps=1e-6
        )
        overlap = (direction * novel).sum(dim=-1)
        torch.testing.assert_close(
            overlap,
            torch.zeros_like(overlap),
            atol=1e-5,
            rtol=0.0,
        )

    def test_all_registered_dual_branch_fusions_keep_one_head(self):
        batch = make_batch()
        for fusion in sorted(DualBranchStateFusion.MODES):
            model = make_model(
                population=False,
                mastery=False,
                transition=False,
                temporal_backbone='mamba_transformer',
                branch_fusion=fusion,
            ).eval()
            with torch.no_grad():
                output = model.forward_sequence(batch, return_aux=True)
            self.assertEqual(output['logits'].shape, (1, 5))
            self.assertTrue(torch.isfinite(output['logits']).all())

    def test_right_padding_does_not_change_valid_predictions(self):
        short = make_batch(length=4, padded_to=4)
        padded = make_batch(length=4, padded_to=7)
        with torch.no_grad():
            expected = self.model.forward_sequence(short)
            actual = self.model.forward_sequence(padded)[:, :3]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_inference_uses_one_integrated_head(self):
        batch = make_batch()
        with torch.no_grad():
            probability = self.model.forward_sequence(batch)
            output = self.model.forward_sequence(batch, return_aux=True)
        torch.testing.assert_close(probability, torch.sigmoid(output['logits']))
        self.assertEqual(output['logits'].shape, (1, 5))
        self.assertEqual(
            set(output['evidence_states']),
            {'population', 'mastery', 'transition'},
        )
        for forbidden in ('long_logits', 'short_logits', 'mastery_logits'):
            self.assertNotIn(forbidden, output)

    def test_all_modules_and_core_receive_task_gradient(self):
        model = make_model().train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.concept_encoder.attribute_encoder[0].weight.grad,
            model.population_encoder.projection.weight.grad,
            model.mastery_evidence.encoder[0].weight.grad,
            model.transition_evidence.edge_projection[0].weight.grad,
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.evidence_fusion.gate[0].weight.grad,
            model.predictor[-1].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_module_ablations_keep_the_same_single_head_contract(self):
        batch = make_batch()
        variants = [
            (False, False, False, False),
            (True, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, True),
            (True, True, True, True),
        ]
        for difficulty, population, mastery, transition in variants:
            model = make_model(
                difficulty, population, mastery, transition
            ).eval()
            with torch.no_grad():
                output = model.forward_sequence(batch, return_aux=True)
            self.assertEqual(output['logits'].shape, (1, 5))
            self.assertTrue(torch.isfinite(output['logits']).all())

        for fusion in ('residual', 'competitive'):
            model = make_model(
                population=False,
                mastery=False,
                concept_memory=True,
                transition=False,
                memory_fusion=fusion,
            ).eval()
            with torch.no_grad():
                output = model.forward_sequence(batch, return_aux=True)
            self.assertEqual(output['logits'].shape, (1, 5))
            self.assertEqual(
                set(output['evidence_states']), {'concept_memory'}
            )

    def test_hard_single_branch_backbones_do_not_construct_the_other_branch(self):
        batch = make_batch()
        cases = (
            ('mamba', 'sequence_encoder', 'short_term'),
            ('transformer', 'short_term', 'sequence_encoder'),
        )
        for backbone, enabled, disabled in cases:
            model = make_model(
                population=False,
                mastery=False,
                transition=False,
                temporal_backbone=backbone,
            ).train()
            self.assertIsNotNone(getattr(model, enabled))
            self.assertIsNone(getattr(model, disabled))
            self.assertIsNone(model.branch_fusion)
            self.assertNotIn(disabled, dict(model.named_modules()))
            output = model.forward_sequence(batch, return_aux=True)
            self.assertEqual(output['logits'].shape, (1, 5))
            output['logits'].sum().backward()
            if backbone == 'mamba':
                gradient = (
                    model.sequence_encoder.layers[0]
                    .cpu_fallback.weight_ih_l0.grad
                )
            else:
                gradient = model.short_term.layers[0].query.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_target_readout_is_a_core_option_not_a_prediction_head(self):
        torch.manual_seed(9)
        plain = make_model(target_readout=False).eval()
        conditioned = make_model(target_readout=True).eval()
        conditioned.load_state_dict(plain.state_dict())
        with torch.no_grad():
            plain_output = plain.forward_sequence(make_batch())
            conditioned_output = conditioned.forward_sequence(make_batch())
        self.assertFalse(torch.allclose(plain_output, conditioned_output))
        self.assertEqual(
            [name for name, _ in conditioned.named_modules() if name == 'predictor'],
            ['predictor'],
        )

    def test_population_backbone_fusion_changes_state_evolution(self):
        torch.manual_seed(17)
        evidence_only = make_model(
            population_fusion='evidence'
        ).eval()
        backbone_only = make_model(
            population_fusion='backbone'
        ).eval()
        backbone_only.load_state_dict(evidence_only.state_dict())
        with torch.no_grad():
            first = evidence_only.forward_sequence(make_batch())
            second = backbone_only.forward_sequence(make_batch())
            auxiliary = backbone_only.forward_sequence(
                make_batch(), return_aux=True
            )
        self.assertFalse(torch.allclose(first, second))
        self.assertNotIn('population', auxiliary['evidence_states'])
        self.assertIsNotNone(
            backbone_only.population_backbone_adapter.last_gate
        )

    def test_population_backbone_adapter_receives_main_loss_gradient(self):
        model = make_model(
            mastery=False,
            transition=False,
            target_readout=False,
            population_fusion='backbone',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.population_encoder.projection.weight.grad,
            model.population_backbone_adapter.delta_projection.weight.grad,
            model.population_backbone_adapter.gate[0].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_population_both_mode_keeps_one_head_and_both_paths(self):
        model = make_model(population_fusion='both').eval()
        with torch.no_grad():
            output = model.forward_sequence(make_batch(), return_aux=True)
        self.assertIn('population', output['evidence_states'])
        self.assertIsNotNone(model.population_backbone_adapter.last_gate)
        self.assertEqual(output['logits'].shape, (1, 5))
        self.assertEqual(
            [name for name, _ in model.named_modules() if name == 'predictor'],
            ['predictor'],
        )

    def test_target_retrieval_reads_prefix_with_one_prediction_head(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
        ).eval()
        with torch.no_grad():
            output = model.forward_sequence(make_batch(), return_aux=True)
        self.assertEqual(set(output['evidence_states']), {'retrieval'})
        self.assertEqual(output['logits'].shape, (1, 5))
        self.assertIsNotNone(model.target_retrieval.last_recency_decay)
        self.assertEqual(
            [name for name, _ in model.named_modules() if name == 'predictor'],
            ['predictor'],
        )

    def test_target_retrieval_preserves_causal_prediction_boundary(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

    def test_target_retrieval_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
            retrieval_source='hybrid',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.target_retrieval.attention.in_proj_weight.grad,
            model.target_retrieval.recency_logit.grad,
            model.target_retrieval.output_encoder[0].weight.grad,
            model.retrieval_memory_projection[0].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_retrieval_source_is_a_general_network_option(self):
        batch = make_batch()
        outputs = []
        for source in ('event', 'state', 'hybrid'):
            model = make_model(
                population=False,
                mastery=False,
                transition=False,
                target_readout=False,
                retrieval=True,
                retrieval_source=source,
            ).eval()
            with torch.no_grad():
                outputs.append(model.forward_sequence(batch))
        for output in outputs:
            self.assertEqual(output.shape, (1, 5))
            self.assertTrue(torch.isfinite(output).all())

    def test_same_scope_retrieval_has_exact_coverage_reliability(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
            retrieval_scope='same',
        ).eval()
        with torch.no_grad():
            output = model.forward_sequence(make_batch(), return_aux=True)
        reliability = output['evidence_reliability']['retrieval']
        self.assertEqual(float(reliability[0, 0, 0]), 0.0)
        self.assertGreater(float(reliability[0, 1, 0]), 0.0)
        self.assertTrue(torch.isfinite(output['logits']).all())

    def test_adaptive_history_router_enforces_cold_local_weight(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
            retrieval_scope='adaptive',
        ).eval()
        with torch.no_grad():
            output = model.forward_sequence(make_batch(), return_aux=True)
        self.assertEqual(
            set(output['evidence_states']),
            {'broad_retrieval', 'local_retrieval'},
        )
        local_weight = output['evidence_gates']['history_local']
        all_weights = torch.cat([
            output['evidence_gates']['history_global'],
            output['evidence_gates']['history_broad'],
            local_weight,
        ], dim=-1)
        self.assertEqual(float(local_weight[0, 0, 0]), 0.0)
        self.assertGreater(float(local_weight[0, 1, 0]), 0.0)
        torch.testing.assert_close(
            all_weights.sum(dim=-1), torch.ones_like(all_weights[..., 0])
        )

    def test_adaptive_history_router_receives_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
            retrieval_scope='adaptive',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.target_retrieval.attention.in_proj_weight.grad,
            model.history_router.broad_projection.weight.grad,
            model.history_router.local_projection.weight.grad,
            model.history_router.router[0].weight.grad,
            model.history_router.router[-1].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rasch_ability_is_one_causal_state_and_one_head(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            ability_mode='evidence',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            output = model.forward_sequence(batch, return_aux=True)
            mutated = model.forward_sequence(changed)
        self.assertEqual(set(output['evidence_states']), {'ability'})
        torch.testing.assert_close(torch.sigmoid(output['logits']), mutated)
        self.assertEqual(
            [name for name, _ in model.named_modules() if name == 'predictor'],
            ['predictor'],
        )

    def test_rasch_ability_moves_with_response_innovation(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            ability_mode='both',
        ).eval()
        correct = make_batch()
        incorrect = copy.deepcopy(correct)
        incorrect['response_seq'][0, 0] = 0.0
        success = 1.0 - model.concept_encoder.difficulty
        with torch.no_grad():
            high = model.rasch_ability.forward_sequence(correct, success)
            low = model.rasch_ability.forward_sequence(incorrect, success)
        self.assertGreater(
            float(high['theta_after'][0, 0]),
            float(low['theta_after'][0, 0]),
        )

    def test_rasch_ability_uses_cross_chunk_prefix_without_student_id(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            ability_mode='evidence',
        ).eval()
        high = make_batch()
        low = make_batch()
        high['initial_concept_attempts'][0, 2] = 10.0
        high['initial_concept_correct'][0, 2] = 9.0
        low['initial_concept_attempts'][0, 2] = 10.0
        low['initial_concept_correct'][0, 2] = 1.0
        high['student_id'] = torch.tensor([17])
        low['student_id'] = torch.tensor([17])
        success = 1.0 - model.concept_encoder.difficulty
        difficulty = model.rasch_ability._concept_difficulty(success)
        with torch.no_grad():
            high_theta, _, _ = model.rasch_ability._initial_ability(
                high, difficulty
            )
            low_theta, _, _ = model.rasch_ability._initial_ability(
                low, difficulty
            )
        self.assertGreater(float(high_theta[0]), float(low_theta[0]))

    def test_rasch_innovation_and_evidence_receive_main_loss_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            ability_mode='both',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.rasch_ability.innovation_encoder[0].weight.grad,
            model.rasch_ability.evidence_encoder[0].weight.grad,
            model.rasch_innovation_adapter.gate[0].weight.grad,
            model.rasch_innovation_adapter.projection.weight.grad,
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.predictor[-1].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_same_scope_retrieval_ignores_unrelated_event_response(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            retrieval=True,
            retrieval_scope='adaptive',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, 1] = 1.0 - changed['response_seq'][0, 1]
        with torch.no_grad():
            original = model.forward_sequence(batch, return_aux=True)
            mutated = model.forward_sequence(changed, return_aux=True)
        torch.testing.assert_close(
            original['evidence_states']['local_retrieval'][:, 3],
            mutated['evidence_states']['local_retrieval'][:, 3],
        )

    def test_factorized_exposure_stream_is_response_free(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            dynamics_mode='factorized',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, 2] = 1.0 - changed['response_seq'][0, 2]
        with torch.no_grad():
            _, concepts, _ = model._concept_views()
            success = model.concept_encoder.success_probability()
            first_events = model._event_context(batch, concepts, success)
            second_events = model._event_context(changed, concepts, success)
            first = model._factorized_states(batch, concepts, first_events)
            second = model._factorized_states(changed, concepts, second_events)
        torch.testing.assert_close(first[0], second[0])
        self.assertFalse(torch.allclose(first[1], second[1]))

    def test_factorized_dynamics_receive_one_head_task_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            dynamics_mode='factorized',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.exposure_sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.acquisition_sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.factorized_fusion.exposure_adapter[0].weight.grad,
            model.factorized_fusion.acquisition_adapter[0].weight.grad,
            model.factorized_fusion.gate[0].weight.grad,
            model.predictor[-1].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(
            [name for name, _ in model.named_modules() if name == 'predictor'],
            ['predictor'],
        )

    def test_factorized_dynamics_preserve_causal_boundary(self):
        model = make_model(
            population=False,
            mastery=False,
            transition=False,
            target_readout=False,
            dynamics_mode='factorized',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

    def test_cold_target_mastery_has_zero_reliability(self):
        with torch.no_grad():
            output = self.model.forward_sequence(
                make_batch(), return_aux=True
            )
        reliability = output['evidence_reliability']['mastery']
        self.assertEqual(float(reliability[0, 0, 0]), 0.0)
        self.assertGreater(float(reliability[0, 1, 0]), 0.0)

    def test_cold_concept_memory_has_zero_reliability_and_gate(self):
        model = make_model(
            population=False,
            mastery=False,
            concept_memory=True,
            transition=False,
        ).eval()
        with torch.no_grad():
            output = model.forward_sequence(make_batch(), return_aux=True)
        reliability = output['evidence_reliability']['concept_memory']
        gate = output['evidence_gates']['concept_memory']
        self.assertEqual(float(reliability[0, 0, 0]), 0.0)
        self.assertEqual(float(gate[0, 0, 0]), 0.0)
        self.assertGreater(float(reliability[0, 1, 0]), 0.0)
        self.assertGreater(float(gate[0, 1, 0]), 0.0)

    def test_unrelated_response_does_not_pollute_target_concept_memory(self):
        model = make_model(
            population=False,
            mastery=False,
            concept_memory=True,
            transition=False,
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, 1] = 1.0 - changed['response_seq'][0, 1]
        with torch.no_grad():
            original = model.forward_sequence(batch, return_aux=True)
            mutated = model.forward_sequence(changed, return_aux=True)
        torch.testing.assert_close(
            original['evidence_states']['concept_memory'][:, 1],
            mutated['evidence_states']['concept_memory'][:, 1],
        )

    def test_dynamic_memory_and_competitive_fusion_receive_gradient(self):
        model = make_model(
            population=False,
            mastery=False,
            concept_memory=True,
            transition=False,
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        output['logits'].sum().backward()
        gradients = [
            model.concept_memory.event_encoder[0].weight.grad,
            model.concept_memory.state_encoder[0].weight.grad,
            model.coverage_fusion.candidate[0].weight.grad,
            model.coverage_fusion.gate[0].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_dynamic_memory_preserves_causal_prediction_boundary(self):
        model = make_model(
            population=False,
            mastery=False,
            concept_memory=True,
            transition=False,
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :-1], mutated[:, :-1])

    def test_difficulty_changes_concept_state_without_new_relation_types(self):
        easy = torch.zeros(9, 2)
        easy[:, 0] = 0.1
        easy[:, 1] = 1.0
        hard = easy.clone()
        hard[3, 0] = 0.9
        first = make_model(statistics=easy).eval()
        second = make_model(statistics=hard).eval()
        second.load_state_dict(first.state_dict(), strict=False)
        second.concept_encoder.difficulty.copy_(hard[:, 0])
        with torch.no_grad():
            first_state = first.concept_encoder()
            second_state = second.concept_encoder()
        self.assertFalse(torch.allclose(first_state[3], second_state[3]))
        self.assertEqual(first.population_encoder.graph.shape[0], 9)
        self.assertEqual(second.population_encoder.graph.shape[0], 9)

    def test_target_count_fallback_is_prefix_only(self):
        attempts, correct = self.model._sequence_target_counts(make_batch())
        torch.testing.assert_close(
            attempts[0], torch.tensor([0., 1., 0., 2., 1.])
        )
        torch.testing.assert_close(
            correct[0], torch.tensor([0., 1., 0., 2., 0.])
        )

    def test_segmented_graph_prefix_matches_naive_cumulative_means(self):
        values = torch.tensor([[[1., 0.], [4., 2.], [3., 2.], [8., 4.]]])
        keys = torch.tensor([[1, 2, 1, 1]])
        valid = torch.tensor([[True, True, True, True]])
        actual = CausalTargetTransitionGraph._segmented_prefix_mean(
            values, keys, valid
        )
        expected = torch.tensor([[[1., 0.], [4., 2.], [2., 1.], [4., 2.]]])
        torch.testing.assert_close(actual, expected)

    def test_recency_graph_prefix_uses_decayed_causal_mean(self):
        values = torch.tensor([[[1.], [9.], [3.], [5.]]])
        keys = torch.tensor([[1, 2, 1, 1]])
        valid = torch.tensor([[True, True, True, True]])
        actual, count = CausalTargetTransitionGraph._segmented_prefix_stats(
            values, keys, valid, decay=0.5
        )
        expected = torch.tensor([[[1.], [9.], [2.6], [53. / 13.]]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            count, torch.tensor([[[1.], [1.], [2.], [3.]]])
        )

    def test_occurrence_recency_decays_by_target_visits_not_event_gap(self):
        values = torch.tensor([[[1.], [9.], [3.], [5.]]])
        keys = torch.tensor([[1, 2, 1, 1]])
        valid = torch.tensor([[True, True, True, True]])
        actual, _ = CausalTargetTransitionGraph._segmented_prefix_stats(
            values,
            keys,
            valid,
            decay=0.5,
            occurrence_weighting=True,
        )
        expected = torch.tensor([
            [[1.], [9.], [7. / 3.], [27. / 7.]]
        ])
        torch.testing.assert_close(actual, expected)

    def test_recency_graph_is_stable_for_full_length_grouped_prefixes(self):
        torch.manual_seed(91)
        batch_size, length, concepts = 8, 500, 40
        values = torch.randn(batch_size, length, 3)
        keys = torch.randint(0, concepts, (batch_size, length))
        keys = keys + torch.arange(batch_size).unsqueeze(1) * concepts
        valid = torch.ones(batch_size, length, dtype=torch.bool)
        actual, _ = CausalTargetTransitionGraph._segmented_prefix_stats(
            values, keys, valid, decay=0.97
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(float(actual.abs().max()), 10.0)

        batch_index, position = batch_size - 1, length - 1
        target = keys[batch_index, position]
        mask = keys[batch_index, :position + 1] == target
        indices = torch.arange(position + 1)[mask]
        weights = 0.97 ** (position - indices.to(torch.float64))
        expected = (
            values[batch_index, :position + 1][mask].to(torch.float64)
            * weights.unsqueeze(1)
        ).sum(dim=0) / weights.sum()
        torch.testing.assert_close(
            actual[batch_index, position].to(torch.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_recency_graph_is_finite_through_length_1000_backward(self):
        torch.manual_seed(92)
        batch_size, length, concepts = 16, 1000, 124
        values = torch.randn(batch_size, length, 3, requires_grad=True)
        keys = torch.randint(0, concepts, (batch_size, length))
        keys = keys + torch.arange(batch_size).unsqueeze(1) * concepts
        valid = torch.ones(batch_size, length, dtype=torch.bool)

        actual, _ = CausalTargetTransitionGraph._segmented_prefix_stats(
            values, keys, valid, decay=0.9
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(float(actual.detach().abs().max()), 10.0)

        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_prefix_calibration_counts_are_strictly_pre_event(self):
        attempts, correct = self.model._event_prefix_counts(make_batch())
        torch.testing.assert_close(
            attempts[0], torch.tensor([0., 0., 1., 0., 2., 1.])
        )
        torch.testing.assert_close(
            correct[0], torch.tensor([0., 0., 1., 0., 2., 0.])
        )

    def test_prefix_calibration_preserves_cold_prior_and_updates_repeats(self):
        torch.manual_seed(47)
        static = make_model(outcome_calibration='static').eval()
        torch.manual_seed(47)
        adaptive = make_model(outcome_calibration='prefix_posterior').eval()
        adaptive.load_state_dict(static.state_dict())
        batch = make_batch()
        attempts, correct = adaptive._event_prefix_counts(batch)
        success = adaptive.concept_encoder.success_probability()
        with torch.no_grad():
            static_outcome = static._outcome_states(
                batch['concept_seq'], batch['response_seq'], success
            )
            adaptive_outcome = adaptive._outcome_states(
                batch['concept_seq'],
                batch['response_seq'],
                success,
                attempts,
                correct,
            )
        torch.testing.assert_close(
            static_outcome[:, :2], adaptive_outcome[:, :2]
        )
        self.assertFalse(torch.allclose(
            static_outcome[:, 2], adaptive_outcome[:, 2]
        ))

    def test_stronger_outcome_prior_is_a_conservative_prefix_update(self):
        torch.manual_seed(52)
        weak = make_model(
            outcome_calibration='prefix_posterior',
            outcome_prior_strength=5.0,
        ).eval()
        torch.manual_seed(52)
        strong = make_model(
            outcome_calibration='prefix_posterior',
            outcome_prior_strength=20.0,
        ).eval()
        torch.manual_seed(52)
        static = make_model(outcome_calibration='static').eval()
        batch = make_batch()
        attempts, correct = weak._event_prefix_counts(batch)
        success = weak.concept_encoder.success_probability()
        with torch.no_grad():
            weak_state = weak._outcome_states(
                batch['concept_seq'], batch['response_seq'], success,
                attempts, correct,
            )
            strong_state = strong._outcome_states(
                batch['concept_seq'], batch['response_seq'], success,
                attempts, correct,
            )
            static_state = static._outcome_states(
                batch['concept_seq'], batch['response_seq'], success
            )
        weak_distance = (weak_state[:, 2] - static_state[:, 2]).norm()
        strong_distance = (strong_state[:, 2] - static_state[:, 2]).norm()
        self.assertLess(float(strong_distance), float(weak_distance))

    def test_prefix_calibration_keeps_current_target_response_out(self):
        model = make_model(
            temporal_backbone='mamba_transformer',
            outcome_calibration='prefix_posterior',
            transition_aggregation='recency',
        ).eval()
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['response_seq'][0, -1] = 1.0 - changed['response_seq'][0, -1]
        with torch.no_grad():
            original = model.forward_sequence(batch)
            mutated = model.forward_sequence(changed)
        torch.testing.assert_close(original, mutated)

    def test_semantic_split_routes_evidence_before_branch_fusion(self):
        model = make_model(
            temporal_backbone='mamba_transformer',
            population=False,
            mastery=False,
            student_graph=True,
            evidence_placement='semantic_split',
            branch_fusion='adaptive_orthogonal_vector',
        ).train()
        batch = make_batch()
        batch['student_id'] = torch.tensor([2])
        output = model.forward_sequence(batch, return_aux=True)
        self.assertIn('student_graph', output['evidence_states'])
        self.assertIn('transition', output['evidence_states'])
        self.assertIn('student_long', output['evidence_gates'])
        self.assertIn('transition_short', output['evidence_gates'])
        self.assertNotIn('student_graph', output['evidence_gates'])
        self.assertNotIn('transition', output['evidence_gates'])
        output['logits'].sum().backward()
        for gradient in (
            model.student_long_adapter.update[0].weight.grad,
            model.transition_short_adapter.update[0].weight.grad,
            model.branch_fusion.vector_router.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_both_placement_keeps_branch_and_posthoc_evidence_paths(self):
        model = make_model(
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            population=False,
            mastery=False,
            student_graph=True,
            evidence_placement='semantic_split_both',
        ).eval()
        batch = make_batch()
        batch['student_id'] = torch.tensor([2])
        with torch.no_grad():
            output = model.forward_sequence(batch, return_aux=True)
        for name in (
            'student_long',
            'transition_short',
            'student_graph',
            'transition',
        ):
            self.assertIn(name, output['evidence_gates'])

    def test_short_memory_sources_preserve_causality_and_one_head(self):
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed['concept_seq'][0, -1] = 7
        changed['response_seq'][0, -1] = 1.0
        predictions = {}
        for source in ('event', 'state', 'mean', 'gated'):
            torch.manual_seed(117)
            model = make_model(
                temporal_backbone='mamba_transformer',
                branch_fusion='adaptive_orthogonal_vector',
                short_memory_source=source,
            ).eval()
            with torch.no_grad():
                original = model.forward_sequence(batch)
                mutated = model.forward_sequence(changed)
            torch.testing.assert_close(original[:, :-1], mutated[:, :-1])
            predictions[source] = original
            self.assertEqual(
                [name for name, _ in model.named_modules() if name == 'predictor'],
                ['predictor'],
            )
        self.assertFalse(torch.allclose(
            predictions['event'], predictions['state']
        ))

    def test_gated_short_memory_and_both_branches_receive_task_gradient(self):
        model = make_model(
            temporal_backbone='mamba_transformer',
            branch_fusion='adaptive_orthogonal_vector',
            short_memory_source='gated',
        ).train()
        output = model.forward_sequence(make_batch(), return_aux=True)
        self.assertIn('short_memory_state', output['evidence_gates'])
        output['logits'].sum().backward()
        for gradient in (
            model.branch_memory_router.gate[0].weight.grad,
            model.sequence_encoder.layers[0].cpu_fallback.weight_ih_l0.grad,
            model.short_term.layers[0].query.weight.grad,
            model.branch_fusion.vector_router.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_placement_variants_preserve_shared_initialization(self):
        kwargs = {
            'temporal_backbone': 'mamba_transformer',
            'population': False,
            'mastery': False,
            'student_graph': True,
        }
        torch.manual_seed(83)
        posthoc = make_model(**kwargs)
        torch.manual_seed(83)
        split = make_model(**kwargs, evidence_placement='semantic_split')
        posthoc_state = posthoc.state_dict()
        split_state = split.state_dict()
        for name, value in posthoc_state.items():
            torch.testing.assert_close(value, split_state[name])


if __name__ == '__main__':
    unittest.main()
