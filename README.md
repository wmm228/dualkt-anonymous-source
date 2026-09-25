# DualKT

Anonymous supplementary implementation of **DualKT: Dual-Timescale Knowledge
Tracing via Orthogonal Decomposition and Target-Conditioned Fusion**.

## Package

- `dualkt/model.py` contains the structured question encoder, dual-timescale
  state construction, orthogonal decomposition, target-conditioned fusion,
  transition refinement, and prediction head.
- `dualkt/layers.py` contains the residual Mamba-2 encoder and hierarchical
  target cross-attention layers used by DualKT.
- `dualkt/__init__.py` exports the public `DualKT` class.

## Environment

The experiments use Python 3.10, PyTorch 2.5.1, CUDA 12.4, Mamba-SSM
2.3.2.post1, and causal-conv1d 1.6.2.post1.

```bash
python -m pip install -r requirements.txt
```

## Model construction

```python
from dualkt import DualKT

model = DualKT(
    n_questions=n_questions,
    n_concepts=n_concepts,
    question_graph=question_graph,
    question_concept_incidence=question_concept_incidence,
    concept_question_incidence=concept_question_incidence,
)
```

The graph and incidence inputs are sparse PyTorch tensors. Index `0` is
reserved for padding. The public constructor fixes the architecture to the
paper configuration: two Mamba-2 layers, two target-query Transformer layers,
a 64-interaction recent window, 32-interaction summary chunks, structured
question encoding, orthogonal target-conditioned fusion, and recency-weighted
transition refinement.

For sequence training, `forward_sequence` consumes a batch containing
`concept_seq`, `question_seq`, `response_seq`, `seq_len`, and
`transition_key_seq`. The transition keys are the exact KC-bundle identifiers
used by the attention bias and transition matching. Calling the model with
`return_aux=True` additionally returns fused states, routing gates, and
evidence diagnostics.
