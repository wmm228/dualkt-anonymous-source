import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as CudaMamba
    from mamba_ssm import Mamba2 as CudaMamba2
except ImportError:
    CudaMamba = None
    CudaMamba2 = None


DEFAULT_DIFFICULTY_DIM = 4
DEFAULT_STUDENT_PROFILE_DIM = 20
RELIABILITY_DIM = 10
HISTORY_NORMALIZATION_LENGTH = 500


def _largest_head_dim(inner_dim, maximum=64):
    for value in range(min(maximum, inner_dim), 0, -1):
        if inner_dim % value == 0:
            return value
    return 1


class SelectiveStateSpaceBlock(nn.Module):
    """CUDA Mamba/Mamba-2 with a GRU fallback for CPU tests."""

    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        version='mamba2',
        layer_idx=0,
    ):
        super().__init__()
        self.version = str(version).strip().lower()
        if self.version not in {'mamba', 'mamba2'}:
            raise ValueError("mamba_version must be either 'mamba' or 'mamba2'")

        self.cuda_layer = None
        if self.version == 'mamba2' and CudaMamba2 is not None:
            self.cuda_layer = CudaMamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=_largest_head_dim(d_model * expand),
                layer_idx=layer_idx,
                sequence_parallel=False,
            )
        elif self.version == 'mamba' and CudaMamba is not None:
            self.cuda_layer = CudaMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
            )
        self.cpu_fallback = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        if self.cuda_layer is not None and x.is_cuda:
            return self.cuda_layer(x)
        output, _ = self.cpu_fallback(x)
        return output


class ResidualMambaEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        n_layers=2,
        dropout=0.2,
        version='mamba2',
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError('mamba_layers must be positive')
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList([
            SelectiveStateSpaceBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                version=version,
                layer_idx=layer_idx,
            )
            for layer_idx in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + self.dropout(layer(norm(x)))
        return self.output_norm(x)


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, time_gaps):
        values = torch.log1p(torch.clamp(time_gaps, min=0.0)) * self.scale
        half_dim = self.d_model // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=values.device) * (-math.log(10000) / denominator)
        )
        angles = values.unsqueeze(-1) * frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.d_model % 2 == 1:
            encoded = F.pad(encoded, (0, 1))
        return encoded


class TargetConditionedLongMamba(nn.Module):
    """Concept-centric full-history state with target-conditioned readout."""

    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        n_layers=2,
        dropout=0.2,
        version='mamba2',
    ):
        super().__init__()
        self.encoder = ResidualMambaEncoder(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            dropout=dropout,
            version=version,
        )
        self.target_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def _read(self, state, target_context):
        gate = self.target_gate(torch.cat([state, target_context], dim=-1))
        return self.output_norm(state * (1.0 + gate))

    def forward(self, x, seq_len, target_context):
        encoded = self.encoder(x)
        last_idx = (seq_len - 1).clamp_min(0)
        batch_idx = torch.arange(x.size(0), device=x.device)
        state = encoded[batch_idx, last_idx]
        return self._read(state, target_context)

    def forward_sequence(self, x, seq_len, target_context):
        if x.size(1) < 2:
            return x.new_zeros(x.size(0), 0, x.size(-1))
        encoded = self.encoder(x)
        state = encoded[:, :-1]
        output = self._read(state, target_context)
        positions = torch.arange(output.size(1), device=x.device).unsqueeze(0)
        valid = positions < (seq_len - 1).unsqueeze(1)
        return output.masked_fill(~valid.unsqueeze(-1), 0.0)


class TargetCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, window_size=64, dropout=0.2):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        distance = torch.arange(window_size - 1, -1, -1, dtype=torch.float32)
        slopes = torch.linspace(0.25, 1.0, n_heads, dtype=torch.float32)
        initial_bias = -slopes.unsqueeze(1) * distance.unsqueeze(0) / max(window_size, 1)
        self.relative_bias = nn.Parameter(initial_bias)

    def forward(self, query, memory, memory_mask, attention_bias=None):
        batch_size, steps, window, d_model = memory.shape
        q = self.query(query).view(batch_size, steps, self.n_heads, self.head_dim)
        k = self.key(memory).view(
            batch_size, steps, window, self.n_heads, self.head_dim
        )
        v = self.value(memory).view(
            batch_size, steps, window, self.n_heads, self.head_dim
        )
        scores = torch.einsum('bshd,bskhd->bshk', q, k) / math.sqrt(self.head_dim)
        scores = scores + self.relative_bias.view(1, 1, self.n_heads, window)
        if attention_bias is not None:
            if attention_bias.shape != memory_mask.shape:
                raise ValueError(
                    'attention_bias must have the same [batch, step, window] '
                    'shape as memory_mask'
                )
            scores = scores + attention_bias.unsqueeze(2).to(scores.dtype)

        safe_mask = memory_mask.clone()
        no_history = ~safe_mask.any(dim=-1)
        if no_history.any():
            safe_mask[..., -1] |= no_history
        scores = scores.masked_fill(~safe_mask.unsqueeze(2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        context = torch.einsum('bshk,bskhd->bshd', weights, v)
        context = context.reshape(batch_size, steps, d_model)
        query = self.norm1(query + self.output_dropout(self.output(context)))
        query = self.norm2(query + self.ffn(query))
        return query, weights


class CausalBlockSummaryAttention(nn.Module):
    """Target-query attention over completed causal history blocks."""

    def __init__(
        self,
        d_model,
        n_heads=4,
        dropout=0.2,
        block_size=32,
        max_seq_len=500,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.max_blocks = math.ceil(max_seq_len / self.block_size)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layer = TargetCrossAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            window_size=self.max_blocks + 1,
            dropout=dropout,
        )
        self.last_attention_weights = None

    def _summaries(self, history, valid_lengths):
        batch_size, length, d_model = history.shape
        blocks = math.ceil(max(length, 1) / self.block_size)
        if blocks > self.max_blocks:
            raise ValueError(
                f'history length {length} exceeds configured block-summary capacity'
            )
        padded_length = blocks * self.block_size
        padded = F.pad(history, (0, 0, 0, padded_length - length))
        positions = torch.arange(padded_length, device=history.device).view(1, -1)
        valid = positions < valid_lengths.view(-1, 1)
        values = padded * valid.unsqueeze(-1)
        values = values.view(batch_size, blocks, self.block_size, d_model)
        counts = valid.view(batch_size, blocks, self.block_size).sum(dim=2)
        summaries = values.sum(dim=2) / counts.clamp_min(1).unsqueeze(-1)
        summary_valid = counts > 0
        if blocks < self.max_blocks:
            summaries = F.pad(summaries, (0, 0, 0, self.max_blocks - blocks))
            summary_valid = F.pad(summary_valid, (0, self.max_blocks - blocks))
        return summaries, summary_valid

    def _memory(self, history, valid_lengths):
        summaries, summary_valid = self._summaries(history, valid_lengths)
        null = self.null_memory.expand(history.size(0), -1, -1)
        return torch.cat([null, summaries], dim=1), summary_valid

    def _sequence_memory(self, history):
        """Build each target step's summaries from history available at that step."""
        batch_size, steps, d_model = history.shape
        memory = history.new_zeros(
            batch_size, steps, self.max_blocks + 1, d_model
        )
        memory[:, :, 0] = self.null_memory
        valid = torch.zeros(
            batch_size,
            steps,
            self.max_blocks + 1,
            dtype=torch.bool,
            device=history.device,
        )
        valid[:, :, 0] = True
        target_steps = torch.arange(steps, device=history.device)
        for block_idx, start in enumerate(range(0, steps, self.block_size)):
            end = min(start + self.block_size, steps)
            block = history[:, start:end]
            cumulative = block.cumsum(dim=1)
            counts = (target_steps - start + 1).clamp(min=0, max=end - start)
            indices = (counts - 1).clamp_min(0)
            summaries = cumulative[:, indices] / counts.clamp_min(1).view(1, -1, 1)
            available = counts > 0
            memory[:, :, block_idx + 1] = summaries.masked_fill(
                ~available.view(1, -1, 1), 0.0
            )
            valid[:, :, block_idx + 1] = available.view(1, -1)
        return memory, valid

    def forward(self, history, seq_len, target_context):
        memory, summary_valid = self._memory(history, seq_len)
        mask = torch.cat([
            torch.ones(history.size(0), 1, dtype=torch.bool, device=history.device),
            summary_valid,
        ], dim=1).unsqueeze(1)
        output, weights = self.layer(
            target_context.unsqueeze(1), memory.unsqueeze(1), mask
        )
        self.last_attention_weights = weights.detach()
        return output.squeeze(1)

    def forward_sequence(self, history, seq_len, target_context):
        memory, mask = self._sequence_memory(history)
        output, weights = self.layer(target_context, memory, mask)
        self.last_attention_weights = weights.detach()
        return output


class TargetCrossAttentionTransformer(nn.Module):
    """Target-query Transformer over a bounded recent-history memory."""

    def __init__(
        self,
        d_model,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        window_size=64,
        summary_block_size=32,
        max_seq_len=500,
        use_semantic_bias=False,
        semantic_bias_strength=0.5,
        semantic_bias_decay=0.9,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError('short_window must be positive')
        self.window_size = int(window_size)
        if semantic_bias_strength < 0.0:
            raise ValueError('semantic_bias_strength must be non-negative')
        if not 0.0 < semantic_bias_decay <= 1.0:
            raise ValueError('semantic_bias_decay must be in (0, 1]')
        self.use_semantic_bias = bool(use_semantic_bias)
        self.semantic_bias_strength = nn.Parameter(
            torch.tensor(float(semantic_bias_strength))
        )
        semantic_distance = torch.arange(
            self.window_size - 1, -1, -1, dtype=torch.float32
        )
        self.register_buffer(
            'semantic_distance_decay',
            float(semantic_bias_decay) ** semantic_distance,
        )
        self.layers = nn.ModuleList([
            TargetCrossAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=self.window_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.summary_attention = (
            CausalBlockSummaryAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                block_size=summary_block_size,
                max_seq_len=max_seq_len,
            )
            if summary_block_size and summary_block_size > 0 else None
        )
        self.hierarchy_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        ) if self.summary_attention is not None else None
        self.hierarchy_norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None
        self.last_summary_attention_weights = None

    def _combine_hierarchies(self, recent, summary, target_context):
        if summary is None:
            return recent
        gate = self.hierarchy_gate(torch.cat([recent, summary, target_context], dim=-1))
        return self.hierarchy_norm(gate * recent + (1.0 - gate) * summary)

    def _sequence_windows(self, history):
        padded = F.pad(history, (0, 0, self.window_size - 1, 0))
        return padded.unfold(1, self.window_size, 1).permute(0, 1, 3, 2)

    def _semantic_bias(self, memory_keys, target_keys):
        if not self.use_semantic_bias:
            return None
        if memory_keys is None or target_keys is None:
            raise ValueError(
                'semantic attention requires history_keys and target_keys'
            )
        match = (
            memory_keys.eq(target_keys.unsqueeze(-1))
            & memory_keys.gt(0)
            & target_keys.unsqueeze(-1).gt(0)
        )
        strength = self.semantic_bias_strength.clamp(min=0.0, max=2.0)
        return (
            match.to(self.semantic_distance_decay.dtype)
            * self.semantic_distance_decay
            * strength
        )

    def forward(
        self,
        history,
        seq_len,
        target_context,
        history_keys=None,
        target_keys=None,
    ):
        batch_size, total_len, _ = history.shape
        offsets = torch.arange(self.window_size, device=history.device) - self.window_size + 1
        end = (seq_len - 1).clamp_min(0)
        positions = end.unsqueeze(1) + offsets.unsqueeze(0)
        safe_positions = positions.clamp(min=0, max=max(total_len - 1, 0))
        batch_idx = torch.arange(batch_size, device=history.device).unsqueeze(1)
        memory = history[batch_idx, safe_positions].unsqueeze(1)
        mask = ((positions >= 0) & (positions < seq_len.unsqueeze(1))).unsqueeze(1)
        semantic_bias = None
        if self.use_semantic_bias:
            if history_keys is None or history_keys.shape != history.shape[:2]:
                raise ValueError('history_keys shape must match history [batch, step]')
            if target_keys is None or target_keys.shape != (batch_size,):
                raise ValueError('target_keys must have shape [batch]')
            memory_keys = history_keys[batch_idx, safe_positions].unsqueeze(1)
            semantic_bias = self._semantic_bias(
                memory_keys, target_keys.unsqueeze(1)
            )
        query = target_context.unsqueeze(1)
        weights = None
        for layer in self.layers:
            query, weights = layer(query, memory, mask, semantic_bias)
        self.last_attention_weights = weights.detach() if weights is not None else None
        recent = query.squeeze(1)
        summary = (
            self.summary_attention(history, seq_len, target_context)
            if self.summary_attention is not None else None
        )
        self.last_summary_attention_weights = (
            self.summary_attention.last_attention_weights
            if self.summary_attention is not None else None
        )
        return self._combine_hierarchies(recent, summary, target_context)

    def forward_sequence(
        self,
        history,
        seq_len,
        target_context,
        history_keys=None,
        target_keys=None,
    ):
        if history.size(1) < 2:
            return history.new_zeros(history.size(0), 0, history.size(-1))
        historical_inputs = history[:, :-1]
        steps = historical_inputs.size(1)
        memory = self._sequence_windows(historical_inputs)
        positions = torch.arange(steps, device=history.device)
        offsets = torch.arange(self.window_size, device=history.device) - self.window_size + 1
        absolute = positions.unsqueeze(1) + offsets.unsqueeze(0)
        mask = (absolute >= 0).unsqueeze(0).expand(history.size(0), -1, -1)
        target_valid = positions.unsqueeze(0) < (seq_len - 1).unsqueeze(1)
        semantic_bias = None
        if self.use_semantic_bias:
            if history_keys is None or history_keys.shape != history.shape[:2]:
                raise ValueError('history_keys shape must match history [batch, step]')
            if target_keys is None or target_keys.shape != (
                history.size(0), steps
            ):
                raise ValueError(
                    'target_keys must have shape [batch, prediction step]'
                )
            memory_keys = self._sequence_windows(
                history_keys[:, :-1].unsqueeze(-1)
            ).squeeze(-1)
            semantic_bias = self._semantic_bias(memory_keys, target_keys)

        query = target_context
        weights = None
        for layer in self.layers:
            query, weights = layer(query, memory, mask, semantic_bias)
        query = query.masked_fill(~target_valid.unsqueeze(-1), 0.0)
        self.last_attention_weights = weights.detach() if weights is not None else None
        summary = (
            self.summary_attention.forward_sequence(
                historical_inputs, seq_len, target_context
            )
            if self.summary_attention is not None else None
        )
        self.last_summary_attention_weights = (
            self.summary_attention.last_attention_weights
            if self.summary_attention is not None else None
        )
        output = self._combine_hierarchies(query, summary, target_context)
        return output.masked_fill(~target_valid.unsqueeze(-1), 0.0)


class ReliabilityAwareFusion(nn.Module):
    def __init__(
        self,
        d_model,
        reliability_dim=RELIABILITY_DIM,
        dropout=0.2,
        rank=None,
        branch_dropout=0.0,
        use_reliability=True,
    ):
        super().__init__()
        rank = rank or max(8, d_model // 4)
        self.branch_dropout = float(branch_dropout)
        gate_dim = d_model * 3 + (reliability_dim if use_reliability else 0)
        self.use_reliability = use_reliability
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.long_projection = nn.Linear(d_model, rank, bias=False)
        self.short_projection = nn.Linear(d_model, rank, bias=False)
        self.interaction_projection = nn.Sequential(
            nn.Linear(rank, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_gate = None

    def _drop_branches(self, h_long, h_short):
        if not self.training or self.branch_dropout <= 0:
            return h_long, h_short
        mask_shape = (*h_long.shape[:-1], 1)
        long_keep = torch.rand(mask_shape, device=h_long.device) >= self.branch_dropout
        short_keep = torch.rand(mask_shape, device=h_short.device) >= self.branch_dropout
        both_dropped = ~long_keep & ~short_keep
        long_keep = long_keep | both_dropped
        return h_long * long_keep, h_short * short_keep

    def forward(self, h_long, h_short, target_context, reliability):
        h_long, h_short = self._drop_branches(h_long, h_short)
        gate_inputs = [h_long, h_short, target_context]
        if self.use_reliability:
            gate_inputs.append(reliability)
        gate = self.gate(torch.cat(gate_inputs, dim=-1))
        interaction = self.interaction_projection(
            self.long_projection(h_long) * self.short_projection(h_short)
        )
        fused = gate * h_long + (1.0 - gate) * h_short + interaction
        self.last_gate = gate.detach()
        return self.output_norm(fused), gate


class TargetConceptMasteryRetriever(nn.Module):
    """Retrieve a target concept's own observed outcomes as a separate state."""

    def __init__(self, d_model, dropout=0.2, max_history=8, lag_buckets=16):
        super().__init__()
        self.max_history = int(max_history)
        self.lag_buckets = int(lag_buckets)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.lag_embedding = nn.Embedding(self.lag_buckets, d_model)
        self.lag_bias = nn.Embedding(self.lag_buckets, 1)
        self.cold_state = nn.Parameter(torch.zeros(d_model))
        self.output = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None

    def forward(self, events, target_context, indices, target_positions):
        squeeze_step = target_context.dim() == 2
        if squeeze_step:
            target_context = target_context.unsqueeze(1)
            indices = indices.unsqueeze(1)
            target_positions = target_positions.unsqueeze(1)

        batch_size, steps, history_size = indices.shape
        if history_size != self.max_history:
            raise ValueError('mastery history has the wrong retrieval width')
        safe_indices = indices.clamp(min=0, max=max(events.size(1) - 1, 0))
        batch_index = torch.arange(
            batch_size, device=events.device
        ).view(batch_size, 1, 1)
        retrieved = events[batch_index, safe_indices]
        valid = indices >= 0

        lag = (target_positions.unsqueeze(-1) - safe_indices).clamp(min=1)
        lag_bucket = torch.floor(torch.log2(lag.float())).long().clamp(
            max=self.lag_buckets - 1
        )
        keys = self.key(retrieved) + self.lag_embedding(lag_bucket)
        query = self.query(target_context).unsqueeze(-2)
        scores = (query * keys).sum(dim=-1) / math.sqrt(events.size(-1))
        scores = scores + self.lag_bias(lag_bucket).squeeze(-1)
        scores = scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        summary = (weights.unsqueeze(-1) * self.value(retrieved)).sum(dim=-2)

        has_history = valid.any(dim=-1)
        cold = self.cold_state.view(1, 1, -1).expand_as(summary)
        summary = torch.where(has_history.unsqueeze(-1), summary, cold)
        count_fraction = valid.float().mean(dim=-1, keepdim=True)
        nearest_lag = torch.where(valid, lag, lag.new_full(lag.shape, 1 << 20)).min(
            dim=-1
        ).values.float()
        recency = torch.where(
            has_history,
            1.0 / torch.log2(nearest_lag + 2.0),
            torch.zeros_like(nearest_lag),
        ).unsqueeze(-1)
        output = self.output(torch.cat([
            summary,
            target_context,
            count_fraction,
            recency,
        ], dim=-1))
        output = self.output_norm(output + summary)
        self.last_attention_weights = weights.detach()
        return output.squeeze(1) if squeeze_step else output


class DualTimescaleConceptStateTracker(nn.Module):
    """Causal fast/slow mastery states over a target concept's own attempts."""

    def __init__(self, d_model, dropout=0.2, max_history=8, lag_buckets=16):
        super().__init__()
        self.max_history = int(max_history)
        self.lag_buckets = int(lag_buckets)
        self.lag_embedding = nn.Embedding(self.lag_buckets, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.fast_cell = nn.GRUCell(d_model, d_model)
        self.slow_cell = nn.GRUCell(d_model, d_model)
        self.fast_cold_state = nn.Parameter(torch.zeros(d_model))
        self.slow_cold_state = nn.Parameter(torch.zeros(d_model))
        self.fast_decay_rate = nn.Parameter(torch.tensor(-1.0))
        self.slow_decay_rate = nn.Parameter(torch.tensor(-3.0))
        self.timescale_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.output = nn.Sequential(
            nn.Linear(d_model * 4 + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None

    @staticmethod
    def _decay(state, cold_state, gap, rate):
        retention = torch.exp(
            -F.softplus(rate) * torch.log1p(gap.float())
        ).unsqueeze(-1)
        return cold_state + retention * (state - cold_state)

    def forward(self, events, target_context, indices, target_positions):
        squeeze_step = target_context.dim() == 2
        if squeeze_step:
            target_context = target_context.unsqueeze(1)
            indices = indices.unsqueeze(1)
            target_positions = target_positions.unsqueeze(1)

        batch_size, steps, history_size = indices.shape
        if history_size != self.max_history:
            raise ValueError('mastery history has the wrong retrieval width')
        safe_indices = indices.clamp(min=0, max=max(events.size(1) - 1, 0))
        batch_index = torch.arange(
            batch_size, device=events.device
        ).view(batch_size, 1, 1)
        retrieved = events[batch_index, safe_indices]
        valid = indices >= 0

        flat_size = batch_size * steps
        fast_cold = self.fast_cold_state.view(1, -1).expand(flat_size, -1)
        slow_cold = self.slow_cold_state.view(1, -1).expand(flat_size, -1)
        fast_state = fast_cold
        slow_state = slow_cold
        last_position = torch.zeros(
            flat_size, dtype=torch.long, device=events.device
        )
        has_history = torch.zeros(
            flat_size, dtype=torch.bool, device=events.device
        )

        flat_indices = safe_indices.reshape(flat_size, history_size)
        flat_valid = valid.reshape(flat_size, history_size)
        flat_events = retrieved.reshape(flat_size, history_size, -1)
        for history_index in range(history_size):
            current_valid = flat_valid[:, history_index]
            current_position = flat_indices[:, history_index]
            gap = torch.where(
                has_history,
                (current_position - last_position).clamp_min(1),
                torch.ones_like(current_position),
            )
            lag_bucket = torch.floor(torch.log2(gap.float())).long().clamp(
                max=self.lag_buckets - 1
            )
            event_input = self.input_norm(
                flat_events[:, history_index] + self.lag_embedding(lag_bucket)
            )
            fast_decayed = self._decay(
                fast_state, fast_cold, gap, self.fast_decay_rate
            )
            slow_decayed = self._decay(
                slow_state, slow_cold, gap, self.slow_decay_rate
            )
            fast_candidate = self.fast_cell(event_input, fast_decayed)
            slow_candidate = self.slow_cell(event_input, slow_decayed)
            fast_state = torch.where(
                current_valid.unsqueeze(-1), fast_candidate, fast_state
            )
            slow_state = torch.where(
                current_valid.unsqueeze(-1), slow_candidate, slow_state
            )
            last_position = torch.where(
                current_valid, current_position, last_position
            )
            has_history = has_history | current_valid

        flat_target_positions = target_positions.reshape(flat_size)
        target_gap = (flat_target_positions - last_position).clamp_min(1)
        fast_state = self._decay(
            fast_state, fast_cold, target_gap, self.fast_decay_rate
        )
        slow_state = self._decay(
            slow_state, slow_cold, target_gap, self.slow_decay_rate
        )
        fast_state = torch.where(
            has_history.unsqueeze(-1), fast_state, fast_cold
        )
        slow_state = torch.where(
            has_history.unsqueeze(-1), slow_state, slow_cold
        )

        flat_target = target_context.reshape(flat_size, -1)
        timescale_weights = torch.softmax(self.timescale_gate(torch.cat([
            fast_state,
            slow_state,
            flat_target,
        ], dim=-1)), dim=-1)
        mixed_state = (
            timescale_weights[:, :1] * fast_state
            + timescale_weights[:, 1:] * slow_state
        )
        count_fraction = flat_valid.float().mean(dim=-1, keepdim=True)
        recency = torch.where(
            has_history,
            1.0 / torch.log2(target_gap.float() + 2.0),
            torch.zeros_like(target_gap, dtype=torch.float),
        ).unsqueeze(-1)
        output = self.output(torch.cat([
            fast_state,
            slow_state,
            mixed_state,
            flat_target,
            count_fraction,
            recency,
        ], dim=-1))
        output = self.output_norm(output + mixed_state)
        output = output.view(batch_size, steps, -1)
        self.last_attention_weights = timescale_weights.detach().view(
            batch_size, steps, 2
        )
        return output.squeeze(1) if squeeze_step else output


class CausalSparseHistoryRetriever(nn.Module):
    """Target-conditioned Top-K readout over all causally available events."""

    def __init__(
        self,
        d_model,
        dropout=0.2,
        max_history=8,
        topk=32,
        query_chunk_size=128,
    ):
        super().__init__()
        del max_history
        self.topk = int(topk)
        self.query_chunk_size = int(query_chunk_size)
        if self.topk < 1 or self.query_chunk_size < 1:
            raise ValueError('semantic top-k and query chunk size must be positive')
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.recency_rate = nn.Parameter(torch.tensor(-3.0))
        self.output = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None
        self.last_selected_indices = None

    def forward(self, events, target_context, indices, target_positions):
        del indices
        squeeze_step = target_context.dim() == 2
        if squeeze_step:
            target_context = target_context.unsqueeze(1)
            target_positions = target_positions.unsqueeze(1)

        batch_size, steps, d_model = target_context.shape
        history_length = events.size(1)
        selected_count = min(self.topk, history_length)
        keys = F.normalize(self.key(events), dim=-1)
        values = self.value(events)
        queries = F.normalize(self.query(target_context), dim=-1)
        history_positions = torch.arange(
            history_length, device=events.device
        ).view(1, 1, history_length)
        batch_index = torch.arange(
            batch_size, device=events.device
        ).view(batch_size, 1, 1)

        outputs = []
        all_weights = []
        all_indices = []
        for start in range(0, steps, self.query_chunk_size):
            end = min(start + self.query_chunk_size, steps)
            chunk_queries = queries[:, start:end]
            chunk_positions = target_positions[:, start:end].unsqueeze(-1)
            scores = torch.einsum(
                'bcd,btd->bct', chunk_queries, keys
            ) / math.sqrt(d_model)
            lag = (chunk_positions - history_positions).clamp_min(1)
            scores = scores - F.softplus(self.recency_rate) * torch.log1p(
                lag.float()
            )
            causal = history_positions < chunk_positions
            scores = scores.masked_fill(
                ~causal, torch.finfo(scores.dtype).min
            )
            top_scores, top_indices = torch.topk(
                scores, k=selected_count, dim=-1
            )
            weights = torch.softmax(top_scores, dim=-1)
            selected_values = values[batch_index, top_indices]
            summary = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
            selected_lag = lag.expand(-1, -1, -1).gather(
                -1, top_indices
            ).float()
            recency = (
                weights / torch.log2(selected_lag + 2.0)
            ).sum(dim=-1, keepdim=True)
            coverage = (
                torch.log1p(chunk_positions.squeeze(-1).float())
                / math.log1p(HISTORY_NORMALIZATION_LENGTH)
            ).clamp(max=1.0).unsqueeze(-1)
            chunk_target = target_context[:, start:end]
            output = self.output(torch.cat([
                summary,
                chunk_target,
                coverage,
                recency,
            ], dim=-1))
            outputs.append(self.output_norm(output + summary))
            all_weights.append(weights.detach())
            all_indices.append(top_indices.detach())

        output = torch.cat(outputs, dim=1)
        self.last_attention_weights = torch.cat(all_weights, dim=1)
        self.last_selected_indices = torch.cat(all_indices, dim=1)
        return output.squeeze(1) if squeeze_step else output


class CalibratedExpertFusion(nn.Module):
    """Route four structurally distinct KT experts using state and evidence."""

    def __init__(
        self,
        d_model,
        reliability_dim=RELIABILITY_DIM,
        dropout=0.2,
        rank=None,
        use_expert_evidence=False,
        use_cross_residual=False,
    ):
        super().__init__()
        rank = rank or max(8, d_model // 4)
        self.use_expert_evidence = bool(use_expert_evidence)
        self.use_cross_residual = bool(use_cross_residual)
        evidence_dim = 10 if self.use_expert_evidence else 0
        self.state_router = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + reliability_dim + evidence_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.cold_prior_strength = nn.Parameter(torch.tensor(1.0))
        self.repeat_long_strength = nn.Parameter(torch.tensor(0.5))
        self.repeat_mastery_strength = nn.Parameter(torch.tensor(1.0))
        self.long_projection = nn.Linear(d_model, rank, bias=False)
        self.short_projection = nn.Linear(d_model, rank, bias=False)
        if self.use_cross_residual:
            self.mastery_projection = nn.Linear(d_model, rank, bias=False)
            self.correction = nn.Sequential(
                nn.Linear(
                    rank * 3 + d_model + reliability_dim + 10,
                    d_model,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Tanh(),
            )
            self.correction_scale = nn.Parameter(torch.tensor(-1.3862944))
        else:
            self.mastery_projection = None
            self.correction = nn.Sequential(
                nn.Linear(rank + d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Tanh(),
            )
            self.correction_scale = None
        self.last_gate = None

    def forward(
        self,
        prior_logits,
        long_logits,
        short_logits,
        mastery_logits,
        h_long,
        h_short,
        h_mastery,
        target_context,
        reliability,
    ):
        experts = torch.stack([
            prior_logits,
            long_logits,
            short_logits,
            mastery_logits,
        ], dim=-1)
        state_signal = self.state_router(torch.cat([
            h_long,
            h_short,
            h_mastery,
        ], dim=-1))
        gate_inputs = [target_context, reliability, state_signal]
        if self.use_expert_evidence:
            scaled = torch.tanh(experts.detach() / 4.0)
            disagreements = torch.stack(
                [
                    (scaled[..., left] - scaled[..., right]).abs()
                    for left in range(4)
                    for right in range(left + 1, 4)
                ],
                dim=-1,
            )
            gate_inputs.extend([scaled, disagreements])
        gate_logits = self.gate(torch.cat(gate_inputs, dim=-1))
        concept_seen = reliability[..., 5].clamp(0.0, 1.0)
        availability_bias = torch.stack([
            (1.0 - concept_seen) * self.cold_prior_strength,
            concept_seen * self.repeat_long_strength,
            torch.zeros_like(concept_seen),
            concept_seen * self.repeat_mastery_strength,
        ], dim=-1)
        gate = torch.softmax(gate_logits + availability_bias, dim=-1)
        long_state = self.long_projection(h_long)
        short_state = self.short_projection(h_short)
        if self.use_cross_residual:
            mastery_state = self.mastery_projection(h_mastery)
            scaled = torch.tanh(experts / 4.0)
            disagreements = torch.stack(
                [
                    (scaled[..., left] - scaled[..., right]).abs()
                    for left in range(4)
                    for right in range(left + 1, 4)
                ],
                dim=-1,
            )
            correction_inputs = torch.cat([
                long_state * short_state,
                long_state * mastery_state,
                short_state * mastery_state,
                target_context,
                reliability,
                scaled,
                disagreements,
            ], dim=-1)
            correction_scale = 0.5 * torch.sigmoid(self.correction_scale)
        else:
            correction_inputs = torch.cat([
                long_state * short_state,
                target_context,
            ], dim=-1)
            correction_scale = 0.1
        correction = correction_scale * self.correction(
            correction_inputs
        ).squeeze(-1)
        logits = torch.sum(gate * experts, dim=-1) + correction
        self.last_gate = gate.detach()
        return logits, gate


class BilinearFusion(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, h_long, h_short, target_context, reliability):
        del target_context, reliability
        interaction = self.bilinear(h_long, h_short)
        return self.output_norm(self.projection(interaction + h_long + h_short)), None


class V5ExpertMaTra4LS_KT(nn.Module):
    """V5: long/short encoders plus explicit target-concept state retrieval."""

    def __init__(
        self,
        n_questions,
        n_concepts,
        d_model=128,
        d_state=32,
        d_conv=4,
        expand=2,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        difficulty_dim=DEFAULT_DIFFICULTY_DIM,
        student_profile_dim=DEFAULT_STUDENT_PROFILE_DIM,
        task_mode='question',
        local_windows=(64,),
        fusion_type='expert',
        mamba_version='mamba2',
        mamba_layers=2,
        short_window=None,
        fusion_rank=None,
        branch_dropout=0.1,
        auxiliary_heads=False,
        summary_block_size=32,
        max_seq_len=500,
        fusion_evidence=False,
        cross_chunk_memory=False,
        structured_profile_residual=False,
        mastery_history_size=8,
        mastery_state_mode='attention',
        fusion_cross_residual=False,
        mastery_separate_encoder=False,
        semantic_topk=32,
        **unused,
    ):
        super().__init__()
        del unused
        self.task_mode = str(task_mode).strip().lower()
        if self.task_mode not in {'concept', 'question'}:
            raise ValueError("task_mode must be either 'concept' or 'question'")
        if self.task_mode == 'question' and n_questions < 1:
            raise ValueError('question-level mode requires a non-empty question vocabulary')
        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.d_model = d_model
        self.difficulty_dim = difficulty_dim
        self.student_profile_dim = student_profile_dim
        self.structured_profile_residual = bool(structured_profile_residual)
        self.base_profile_dim = (
            min(student_profile_dim, 13)
            if self.structured_profile_residual else student_profile_dim
        )
        self.structured_profile_dim = (
            max(student_profile_dim - 13, 0)
            if self.structured_profile_residual else 0
        )
        self.reliability_dim = 10 if student_profile_dim >= 13 else 7
        self.cross_chunk_memory = bool(cross_chunk_memory)
        self.mastery_history_size = int(mastery_history_size)
        self.mastery_state_mode = str(mastery_state_mode).strip().lower()
        if self.mastery_state_mode not in {
            'attention', 'dual_state', 'semantic_sparse'
        }:
            raise ValueError(
                "mastery_state_mode must be 'attention', 'dual_state', or "
                "'semantic_sparse'"
            )
        self.mastery_separate_encoder = bool(mastery_separate_encoder)
        self.fusion_type = str(fusion_type).strip().lower()
        self.auxiliary_heads = bool(auxiliary_heads)
        if self.fusion_type not in {'expert', 'reliability', 'gated', 'bilinear'}:
            raise ValueError(
                "fusion_type must be 'expert', 'reliability', 'gated', or 'bilinear'"
            )

        if short_window is None:
            values = list(local_windows) if local_windows is not None else [64]
            short_window = max(int(value) for value in values)
        self.short_window = int(short_window)

        self.question_emb = (
            nn.Embedding(n_questions, d_model) if self.task_mode == 'question' else None
        )
        self.concept_emb = nn.Embedding(n_concepts, d_model)
        self.response_emb = nn.Embedding(2, d_model)
        self.concept_interaction_emb = nn.Embedding(n_concepts * 2, d_model)
        self.question_interaction_emb = (
            nn.Embedding(n_questions * 2, d_model)
            if self.task_mode == 'question' else None
        )
        self.time_encoder = TimeEncoding(d_model)
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(difficulty_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(self.base_profile_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.structured_profile_encoder = None
        self.structured_profile_strength = None
        self.mastery_memory_encoder = None
        self.mastery_memory_strength = None
        self.long_input_fusion = nn.Sequential(
            nn.Linear(d_model * 6, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        short_components = 7 if self.task_mode == 'question' else 6
        self.short_input_fusion = nn.Sequential(
            nn.Linear(d_model * short_components, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.mastery_event_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.mastery_concept_emb = None
        self.mastery_response_emb = None
        self.mastery_concept_interaction_emb = None
        self.mastery_difficulty_encoder = None
        self.mastery_profile_encoder = None
        self.mastery_question_emb = None
        self.mastery_target_context_norm = None
        self.mastery_concept_bias = None
        if self.mastery_separate_encoder:
            self.mastery_concept_emb = nn.Embedding(n_concepts, d_model)
            self.mastery_response_emb = nn.Embedding(2, d_model)
            self.mastery_concept_interaction_emb = nn.Embedding(
                n_concepts * 2, d_model
            )
            self.mastery_difficulty_encoder = nn.Sequential(
                nn.Linear(difficulty_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.mastery_profile_encoder = nn.Sequential(
                nn.Linear(student_profile_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.mastery_question_emb = (
                nn.Embedding(n_questions, d_model)
                if self.task_mode == 'question' else None
            )
            self.mastery_target_context_norm = nn.LayerNorm(d_model)
            self.mastery_concept_bias = nn.Embedding(n_concepts, 1)

        self.long_term = TargetConditionedLongMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=mamba_layers,
            dropout=dropout,
            version=mamba_version,
        )
        self.short_term = TargetCrossAttentionTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            window_size=self.short_window,
            summary_block_size=summary_block_size,
            max_seq_len=max_seq_len,
        )
        mastery_retriever_class = {
            'attention': TargetConceptMasteryRetriever,
            'dual_state': DualTimescaleConceptStateTracker,
            'semantic_sparse': CausalSparseHistoryRetriever,
        }[self.mastery_state_mode]
        retriever_arguments = dict(
            d_model=d_model,
            dropout=dropout,
            max_history=self.mastery_history_size,
        )
        if self.mastery_state_mode == 'semantic_sparse':
            retriever_arguments['topk'] = semantic_topk
        self.mastery_retriever = mastery_retriever_class(**retriever_arguments)
        if self.fusion_type == 'expert':
            self.fusion = CalibratedExpertFusion(
                d_model=d_model,
                reliability_dim=self.reliability_dim,
                dropout=dropout,
                rank=fusion_rank,
                use_expert_evidence=fusion_evidence,
                use_cross_residual=fusion_cross_residual,
            )
        elif self.fusion_type == 'bilinear':
            self.fusion = BilinearFusion(d_model, dropout=dropout)
        else:
            self.fusion = ReliabilityAwareFusion(
                d_model=d_model,
                reliability_dim=self.reliability_dim,
                dropout=dropout,
                rank=fusion_rank,
                branch_dropout=branch_dropout,
                use_reliability=self.fusion_type == 'reliability',
            )
        self.target_context_norm = nn.LayerNorm(d_model)
        self.concept_bias = nn.Embedding(n_concepts, 1)

        prediction_components = 5 if self.task_mode == 'question' else 4

        def make_predictor():
            return nn.Sequential(
                nn.Linear(d_model * prediction_components, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        self.predictor = make_predictor() if self.fusion_type != 'expert' else None
        build_experts = self.fusion_type == 'expert' or self.auxiliary_heads
        self.long_predictor = make_predictor() if build_experts else None
        self.short_predictor = make_predictor() if build_experts else None
        self.mastery_predictor = make_predictor() if build_experts else None
        self.prior_predictor = (
            nn.Sequential(
                nn.Linear(d_model * (prediction_components - 1), d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.fusion_type == 'expert' else None
        )
        self._init_weights()
        self._init_context_residuals()

    def _init_context_residuals(self):
        if self.structured_profile_dim:
            self.structured_profile_encoder = nn.Sequential(
                nn.Linear(self.structured_profile_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.structured_profile_strength = nn.Parameter(torch.tensor(-4.0))
            nn.init.xavier_uniform_(self.structured_profile_encoder[0].weight)
            nn.init.zeros_(self.structured_profile_encoder[0].bias)
            nn.init.zeros_(self.structured_profile_encoder[2].weight)
            nn.init.zeros_(self.structured_profile_encoder[2].bias)

        self.mastery_memory_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.mastery_memory_strength = nn.Parameter(torch.tensor(-4.0))
        nn.init.xavier_uniform_(self.mastery_memory_encoder[0].weight)
        nn.init.zeros_(self.mastery_memory_encoder[0].bias)
        nn.init.zeros_(self.mastery_memory_encoder[2].weight)
        nn.init.zeros_(self.mastery_memory_encoder[2].bias)

    def _match_feature_dim(self, feature, target_dim):
        if feature.size(-1) == target_dim:
            return feature.float()
        if feature.size(-1) > target_dim:
            return feature[..., :target_dim].float()
        return F.pad(feature.float(), (0, target_dim - feature.size(-1)))

    def _feature_or_zeros(self, feature, *shape, target_dim, device):
        if feature is None:
            return torch.zeros(*shape, target_dim, device=device)
        return self._match_feature_dim(feature.to(device), target_dim)

    def _embed_difficulty(self, difficulty_features):
        return self.difficulty_encoder(
            self._match_feature_dim(difficulty_features, self.difficulty_dim)
        )

    def _embed_profile(self, student_profile):
        profile = self._match_feature_dim(
            student_profile, self.student_profile_dim
        )
        embedded = self.profile_encoder(profile[..., :self.base_profile_dim])
        if self.structured_profile_encoder is not None:
            structured = self.structured_profile_encoder(
                profile[..., self.base_profile_dim:]
            )
            embedded = embedded + torch.sigmoid(
                self.structured_profile_strength
            ) * structured
        return embedded

    def _embed_interactions(
        self,
        question_seq,
        concept_seq,
        response_seq,
        time_gap_seq,
        difficulty_seq=None,
        student_profile_seq=None,
        mastery_memory=None,
    ):
        del mastery_memory
        response_seq = response_seq.long().clamp(min=0, max=1)
        c_emb = self.concept_emb(concept_seq)
        r_emb = self.response_emb(response_seq)
        concept_interaction = self.concept_interaction_emb(
            concept_seq + response_seq * self.n_concepts
        )
        time_emb = self.time_encoder(time_gap_seq)
        batch_size, seq_len = concept_seq.shape
        difficulty_seq = self._feature_or_zeros(
            difficulty_seq,
            batch_size,
            seq_len,
            target_dim=self.difficulty_dim,
            device=concept_seq.device,
        )
        student_profile_seq = self._feature_or_zeros(
            student_profile_seq,
            batch_size,
            seq_len,
            target_dim=self.student_profile_dim,
            device=concept_seq.device,
        )
        profile_emb = self._embed_profile(student_profile_seq)

        long_difficulty = difficulty_seq.clone()
        if long_difficulty.size(-1) >= 4:
            long_difficulty[..., 0] = 0.0
            long_difficulty[..., 2] = 0.0
        long_components = [
            c_emb,
            r_emb,
            concept_interaction,
            time_emb,
            self._embed_difficulty(long_difficulty),
            profile_emb,
        ]
        if self.task_mode == 'question':
            if question_seq is None:
                raise ValueError('question-level mode requires question_seq in every batch')
            q_emb = self.question_emb(question_seq)
            question_interaction = self.question_interaction_emb(
                question_seq + response_seq * self.n_questions
            )
            short_components = [
                q_emb,
                c_emb,
                r_emb,
                question_interaction,
                time_emb,
                self._embed_difficulty(difficulty_seq),
                profile_emb,
            ]
        else:
            short_components = [
                c_emb,
                r_emb,
                concept_interaction,
                time_emb,
                self._embed_difficulty(difficulty_seq),
                profile_emb,
            ]
        return (
            self.long_input_fusion(torch.cat(long_components, dim=-1)),
            self.short_input_fusion(torch.cat(short_components, dim=-1)),
        )

    def _mastery_event_states(
        self,
        concept_seq,
        response_seq,
        student_profile_seq,
    ):
        response = response_seq.long().clamp(min=0, max=1)
        profile = self._feature_or_zeros(
            student_profile_seq,
            concept_seq.size(0),
            concept_seq.size(1),
            target_dim=self.student_profile_dim,
            device=concept_seq.device,
        )
        if self.mastery_separate_encoder:
            components = [
                self.mastery_concept_interaction_emb(
                    concept_seq + response * self.n_concepts
                ),
                self.mastery_response_emb(response),
                self.mastery_profile_encoder(profile),
            ]
        else:
            components = [
                self.concept_interaction_emb(
                    concept_seq + response * self.n_concepts
                ),
                self.response_emb(response),
                self._embed_profile(profile),
            ]
        return self.mastery_event_fusion(torch.cat(components, dim=-1))

    def _mastery_target_representation(
        self,
        target_question,
        target_concept,
        target_difficulty,
        target_profile,
        shared_features,
        shared_context,
    ):
        if not self.mastery_separate_encoder:
            return shared_features, shared_context
        features = []
        if self.task_mode == 'question':
            if target_question is None:
                raise ValueError('question-level mode requires target question IDs')
            features.append(self.mastery_question_emb(target_question))
        features.extend([
            self.mastery_concept_emb(target_concept),
            self.mastery_difficulty_encoder(
                self._match_feature_dim(target_difficulty, self.difficulty_dim)
            ),
            self.mastery_profile_encoder(
                self._match_feature_dim(target_profile, self.student_profile_dim)
            ),
        ])
        context = self.mastery_target_context_norm(
            torch.stack(features, dim=0).sum(dim=0)
        )
        return features, context

    def _mastery_prediction_logits(
        self,
        state,
        target_features,
        target_concept,
    ):
        logits = self.mastery_predictor(
            torch.cat([state, *target_features], dim=-1)
        ).squeeze(-1)
        bias = (
            self.mastery_concept_bias(target_concept).squeeze(-1)
            if self.mastery_separate_encoder else
            self.concept_bias(target_concept).squeeze(-1)
        )
        return logits + bias

    def _fallback_sequence_mastery_indices(self, concept_seq):
        batch_size, length = concept_seq.shape
        output = torch.full(
            (batch_size, max(length - 1, 0), self.mastery_history_size),
            -1,
            dtype=torch.long,
            device=concept_seq.device,
        )
        for batch_index, values in enumerate(concept_seq.detach().cpu().tolist()):
            seen = {}
            if not values:
                continue
            seen.setdefault(values[0], []).append(0)
            for target_position in range(1, length):
                history = seen.get(values[target_position], [])[
                    -self.mastery_history_size:
                ]
                if history:
                    output[
                        batch_index,
                        target_position - 1,
                        :len(history),
                    ] = torch.tensor(history, device=concept_seq.device)
                seen.setdefault(values[target_position], []).append(target_position)
        return output

    def _fallback_target_mastery_indices(self, concept_seq, target_concept):
        output = torch.full(
            (concept_seq.size(0), self.mastery_history_size),
            -1,
            dtype=torch.long,
            device=concept_seq.device,
        )
        for batch_index, values in enumerate(concept_seq.detach().cpu().tolist()):
            target = int(target_concept[batch_index])
            history = [
                index for index, concept in enumerate(values)
                if int(concept) == target
            ][-self.mastery_history_size:]
            if history:
                output[batch_index, :len(history)] = torch.tensor(
                    history, device=concept_seq.device
                )
        return output

    def _target_mastery_memory(self, batch, target_concept):
        if not self.cross_chunk_memory:
            return None
        batch_size = target_concept.size(0)
        attempts = batch.get('initial_concept_attempts')
        correct = batch.get('initial_concept_correct')
        if attempts is None or correct is None:
            attempts = torch.zeros(
                batch_size, self.n_concepts, device=target_concept.device
            )
            correct = torch.zeros_like(attempts)
        else:
            attempts = attempts.to(target_concept.device).float()
            correct = correct.to(target_concept.device).float()
        if attempts.size(-1) != self.n_concepts:
            raise ValueError('initial concept memory has the wrong vocabulary size')

        target_shape = target_concept.shape
        flat_target = target_concept.reshape(batch_size, -1)
        target_attempts = attempts.gather(1, flat_target).reshape(target_shape)
        target_correct = correct.gather(1, flat_target).reshape(target_shape)
        prior_strength = 5.0
        mastery = (target_correct + 0.5 * prior_strength) / (
            target_attempts + prior_strength
        )
        confidence = torch.log1p(target_attempts) / math.log1p(
            HISTORY_NORMALIZATION_LENGTH
        )
        memory_features = torch.stack([
            2.0 * (mastery - 0.5) * confidence,
            confidence,
        ], dim=-1)
        strength = torch.sigmoid(self.mastery_memory_strength)
        return strength * self.mastery_memory_encoder(memory_features)

    def _target_representation(
        self,
        target_question,
        target_concept,
        target_difficulty,
        target_profile,
        target_memory=None,
    ):
        features = []
        if self.task_mode == 'question':
            if target_question is None:
                raise ValueError('question-level mode requires target question IDs')
            features.append(self.question_emb(target_question))
        features.extend([
            self.concept_emb(target_concept),
            self._embed_difficulty(target_difficulty),
            self._embed_profile(target_profile),
        ])
        context = torch.stack(features, dim=0).sum(dim=0)
        if target_memory is not None:
            context = context + target_memory
        context = self.target_context_norm(context)
        return features, context

    def _reliability_features(
        self,
        history_fraction,
        target_difficulty,
        target_profile,
    ):
        profile = self._match_feature_dim(target_profile, self.student_profile_dim)
        difficulty = self._match_feature_dim(target_difficulty, self.difficulty_dim)
        attempts = profile[..., 2] if profile.size(-1) > 2 else torch.zeros_like(history_fraction)
        concept_attempts = (
            profile[..., 6] if profile.size(-1) > 6 else torch.zeros_like(history_fraction)
        )
        last_gap = profile[..., 7] if profile.size(-1) > 7 else torch.zeros_like(history_fraction)
        frequency_index = 2 if self.task_mode == 'question' else 3
        target_frequency = (
            difficulty[..., frequency_index]
            if difficulty.size(-1) > frequency_index else torch.zeros_like(history_fraction)
        )
        concept_seen = (
            profile[..., 8] if profile.size(-1) > 8 else (concept_attempts > 0).float()
        )
        concept_lag = (
            profile[..., 9] if profile.size(-1) > 9 else torch.ones_like(history_fraction)
        )
        last_concept_response = (
            profile[..., 10] if profile.size(-1) > 10
            else torch.full_like(history_fraction, 0.5)
        )
        concept_recent_accuracy = (
            profile[..., 11] if profile.size(-1) > 11 else profile[..., 1]
        )
        concept_ema = profile[..., 12] if profile.size(-1) > 12 else concept_recent_accuracy
        features = [
            history_fraction,
            attempts,
            concept_attempts,
            last_gap,
            target_frequency,
            concept_seen,
            concept_lag,
        ]
        if self.reliability_dim > 7:
            features.extend([
                last_concept_response,
                concept_recent_accuracy,
                concept_ema,
            ])
        return torch.stack(features, dim=-1).clamp(min=0.0, max=1.0)

    def _prediction_logits(self, state, target_features, predictor, target_concept):
        logits = predictor(torch.cat([state, *target_features], dim=-1)).squeeze(-1)
        return logits + self.concept_bias(target_concept).squeeze(-1)

    def _prior_logits(self, target_features, target_concept):
        logits = self.prior_predictor(torch.cat(target_features, dim=-1)).squeeze(-1)
        return logits + self.concept_bias(target_concept).squeeze(-1)

    def _init_weights(self):
        for name, module in self.named_modules():
            # Mamba has specialized dt/A initializers that must be preserved.
            if '.cuda_layer' in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _outputs(
        self,
        logits,
        h_long,
        h_short,
        h_mastery,
        target_features,
        target_concept,
        gate,
        expert_logits=None,
    ):
        outputs = {
            'logits': logits,
            'fusion_gate': gate,
            'attention_weights': self.short_term.last_attention_weights,
            'summary_attention_weights': self.short_term.last_summary_attention_weights,
        }
        if expert_logits is not None:
            outputs.update(expert_logits)
        elif self.auxiliary_heads:
            outputs['long_logits'] = self._prediction_logits(
                h_long, target_features, self.long_predictor, target_concept
            )
            outputs['short_logits'] = self._prediction_logits(
                h_short, target_features, self.short_predictor, target_concept
            )
            outputs['mastery_logits'] = self._prediction_logits(
                h_mastery, target_features, self.mastery_predictor, target_concept
            )
        return outputs

    def forward(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        question_seq = batch.get('question_seq')
        concept_seq = batch['concept_seq']
        response_seq = batch['response_seq']
        seq_len = batch['seq_len']
        batch_size, total_len = concept_seq.shape
        difficulty_seq = self._feature_or_zeros(
            batch.get('difficulty_seq'),
            batch_size,
            total_len,
            target_dim=self.difficulty_dim,
            device=concept_seq.device,
        )
        profile_seq = self._feature_or_zeros(
            batch.get('student_profile_seq'),
            batch_size,
            total_len,
            target_dim=self.student_profile_dim,
            device=concept_seq.device,
        )
        target_difficulty = self._feature_or_zeros(
            batch.get('target_difficulty'),
            batch_size,
            target_dim=self.difficulty_dim,
            device=concept_seq.device,
        )
        target_profile = self._feature_or_zeros(
            batch.get('target_profile'),
            batch_size,
            target_dim=self.student_profile_dim,
            device=concept_seq.device,
        )
        long_x, short_x = self._embed_interactions(
            question_seq,
            concept_seq,
            response_seq,
            batch['time_gap_seq'],
            difficulty_seq,
            profile_seq,
        )
        target_memory = self._target_mastery_memory(
            batch, batch['target_concept']
        )
        target_features, target_context = self._target_representation(
            batch.get('target_question'),
            batch['target_concept'],
            target_difficulty,
            target_profile,
            target_memory,
        )
        mastery_target_features, mastery_target_context = (
            self._mastery_target_representation(
                batch.get('target_question'),
                batch['target_concept'],
                target_difficulty,
                target_profile,
                target_features,
                target_context,
            )
        )
        h_long = self.long_term(long_x, seq_len, target_context)
        h_short = self.short_term(short_x, seq_len, target_context)
        mastery_events = self._mastery_event_states(
            concept_seq, response_seq, profile_seq
        )
        mastery_indices = batch.get('mastery_history_indices')
        if mastery_indices is None:
            mastery_indices = self._fallback_target_mastery_indices(
                concept_seq, batch['target_concept']
            )
        h_mastery = self.mastery_retriever(
            mastery_events,
            mastery_target_context,
            mastery_indices,
            seq_len,
        )
        history_fraction = torch.log1p(seq_len.float()) / math.log1p(
            HISTORY_NORMALIZATION_LENGTH
        )
        reliability = self._reliability_features(
            history_fraction, target_difficulty, target_profile
        )
        expert_logits = None
        if self.fusion_type == 'expert':
            long_logits = self._prediction_logits(
                h_long, target_features, self.long_predictor, batch['target_concept']
            )
            short_logits = self._prediction_logits(
                h_short, target_features, self.short_predictor, batch['target_concept']
            )
            prior_logits = self._prior_logits(
                target_features, batch['target_concept']
            )
            mastery_logits = self._mastery_prediction_logits(
                h_mastery,
                mastery_target_features,
                batch['target_concept'],
            )
            logits, gate = self.fusion(
                prior_logits,
                long_logits,
                short_logits,
                mastery_logits,
                h_long,
                h_short,
                h_mastery,
                target_context,
                reliability,
            )
            expert_logits = {
                'prior_logits': prior_logits,
                'long_logits': long_logits,
                'short_logits': short_logits,
                'mastery_logits': mastery_logits,
            }
        else:
            fused, gate = self.fusion(h_long, h_short, target_context, reliability)
            logits = self._prediction_logits(
                fused, target_features, self.predictor, batch['target_concept']
            )
        if not return_aux:
            return torch.sigmoid(logits)
        return self._outputs(
            logits,
            h_long,
            h_short,
            h_mastery,
            target_features,
            batch['target_concept'],
            gate,
            expert_logits,
        )

    def forward_sequence(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        question_seq = batch.get('question_seq')
        concept_seq = batch['concept_seq']
        response_seq = batch['response_seq']
        seq_len = batch['seq_len']
        batch_size, total_len = concept_seq.shape
        if total_len < 2:
            empty = concept_seq.new_zeros(batch_size, 0, dtype=torch.float)
            return {'logits': empty} if return_aux else empty

        difficulty_seq = self._feature_or_zeros(
            batch.get('difficulty_seq'),
            batch_size,
            total_len,
            target_dim=self.difficulty_dim,
            device=concept_seq.device,
        )
        profile_seq = self._feature_or_zeros(
            batch.get('student_profile_seq'),
            batch_size,
            total_len,
            target_dim=self.student_profile_dim,
            device=concept_seq.device,
        )
        long_x, short_x = self._embed_interactions(
            question_seq,
            concept_seq,
            response_seq,
            batch['time_gap_seq'],
            difficulty_seq,
            profile_seq,
        )
        target_question = question_seq[:, 1:] if question_seq is not None else None
        target_concept = concept_seq[:, 1:]
        target_difficulty = difficulty_seq[:, 1:]
        target_profile = profile_seq[:, 1:]
        target_memory = self._target_mastery_memory(batch, target_concept)
        target_features, target_context = self._target_representation(
            target_question,
            target_concept,
            target_difficulty,
            target_profile,
            target_memory,
        )
        mastery_target_features, mastery_target_context = (
            self._mastery_target_representation(
                target_question,
                target_concept,
                target_difficulty,
                target_profile,
                target_features,
                target_context,
            )
        )
        h_long = self.long_term.forward_sequence(long_x, seq_len, target_context)
        h_short = self.short_term.forward_sequence(short_x, seq_len, target_context)
        mastery_events = self._mastery_event_states(
            concept_seq, response_seq, profile_seq
        )
        mastery_indices = batch.get('mastery_history_indices')
        if mastery_indices is None:
            mastery_indices = self._fallback_sequence_mastery_indices(concept_seq)
        target_positions = torch.arange(
            1, total_len, device=concept_seq.device
        ).view(1, -1).expand(batch_size, -1)
        h_mastery = self.mastery_retriever(
            mastery_events,
            mastery_target_context,
            mastery_indices,
            target_positions,
        )

        steps = total_len - 1
        history_fraction = torch.log1p(
            torch.arange(1, steps + 1, device=concept_seq.device, dtype=torch.float32)
        ) / math.log1p(HISTORY_NORMALIZATION_LENGTH)
        history_fraction = history_fraction.view(1, steps).expand(batch_size, -1)
        reliability = self._reliability_features(
            history_fraction, target_difficulty, target_profile
        )
        expert_logits = None
        if self.fusion_type == 'expert':
            long_logits = self._prediction_logits(
                h_long, target_features, self.long_predictor, target_concept
            )
            short_logits = self._prediction_logits(
                h_short, target_features, self.short_predictor, target_concept
            )
            prior_logits = self._prior_logits(target_features, target_concept)
            mastery_logits = self._mastery_prediction_logits(
                h_mastery,
                mastery_target_features,
                target_concept,
            )
            logits, gate = self.fusion(
                prior_logits,
                long_logits,
                short_logits,
                mastery_logits,
                h_long,
                h_short,
                h_mastery,
                target_context,
                reliability,
            )
            expert_logits = {
                'prior_logits': prior_logits,
                'long_logits': long_logits,
                'short_logits': short_logits,
                'mastery_logits': mastery_logits,
            }
        else:
            fused, gate = self.fusion(h_long, h_short, target_context, reliability)
            logits = self._prediction_logits(
                fused, target_features, self.predictor, target_concept
            )
        if not return_aux:
            return torch.sigmoid(logits)
        return self._outputs(
            logits,
            h_long,
            h_short,
            h_mastery,
            target_features,
            target_concept,
            gate,
            expert_logits,
        )


class MultiRelationConceptEncoder(nn.Module):
    """Learn concept states without collapsing distinct global relations."""

    def __init__(self, relation_tensors, d_model, n_layers=2, dropout=0.2):
        super().__init__()
        if relation_tensors.ndim != 3:
            raise ValueError('relation_tensors must have shape [relations, concepts, concepts]')
        if relation_tensors.size(1) != relation_tensors.size(2):
            raise ValueError('every concept relation must be square')
        self.n_relations = int(relation_tensors.size(0))
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError('graph_layers must be positive')

        for index, relation in enumerate(relation_tensors):
            self.register_buffer(
                f'relation_{index}', relation.float().to_sparse().coalesce()
            )
        self.transforms = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_model, d_model, bias=False)
                for _ in range(self.n_relations)
            ])
            for _ in range(self.n_layers)
        ])
        self.relation_scorers = nn.ModuleList([
            nn.Linear(d_model, 1, bias=False) for _ in range(self.n_layers)
        ])
        self.relation_bias = nn.Parameter(
            torch.zeros(self.n_layers, self.n_relations)
        )
        self.updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(self.n_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.n_layers)
        ])
        self.last_relation_gate = None

    def forward(self, concept_table):
        state = concept_table
        relation_states = None
        gate = None
        for layer in range(self.n_layers):
            propagated = []
            for relation_index, transform in enumerate(self.transforms[layer]):
                relation = getattr(self, f'relation_{relation_index}')
                propagated.append(
                    torch.sparse.mm(relation, transform(state))
                )
            relation_states = torch.stack(propagated, dim=0)
            scores = self.relation_scorers[layer](relation_states).squeeze(-1)
            scores = scores + self.relation_bias[layer].unsqueeze(1)
            gate = torch.softmax(scores, dim=0)
            message = (gate.unsqueeze(-1) * relation_states).sum(dim=0)
            state = self.norms[layer](state + self.updates[layer](message))
            state = self.ffn_norms[layer](state + self.ffns[layer](state))
        self.last_relation_gate = gate.detach()
        return state, relation_states, gate


class StudentConceptBipartiteEncoder(nn.Module):
    """DGMKT-style weighted student-concept hypergraph encoder."""

    def __init__(
        self,
        incidence,
        d_model,
        dropout=0.2,
        trainable_anchor=False,
    ):
        super().__init__()
        if incidence.ndim != 2:
            raise ValueError('student incidence must have shape [students, concepts]')
        incidence = incidence.float()
        self.n_students = int(incidence.size(0))
        self.n_concepts = int(incidence.size(1))
        row_sum = incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer(
            'student_to_concept', (incidence / row_sum).to_sparse().coalesce()
        )
        student_degree = incidence.sum(dim=1).clamp_min(1.0)
        concept_degree = incidence.sum(dim=0).clamp_min(1.0)
        student_scale = student_degree.rsqrt()
        hypergraph_left = (
            student_scale.unsqueeze(1)
            * incidence
            / concept_degree.unsqueeze(0)
        )
        hypergraph_right = incidence.T * student_scale.unsqueeze(0)
        self.register_buffer(
            'hypergraph_left', hypergraph_left.to_sparse().coalesce()
        )
        self.register_buffer(
            'hypergraph_right', hypergraph_right.to_sparse().coalesce()
        )
        anchor = torch.empty(self.n_students, d_model)
        nn.init.normal_(anchor, std=0.02)
        if trainable_anchor:
            self.student_anchor = nn.Parameter(anchor)
        else:
            self.register_buffer('student_anchor', anchor)
        self.content_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, concept_table):
        if concept_table.size(0) != self.n_concepts:
            raise ValueError('student incidence and concept vocabulary do not match')
        content = torch.sparse.mm(self.student_to_concept, concept_table)
        concept_anchor = torch.sparse.mm(self.hypergraph_right, self.student_anchor)
        structural = torch.sparse.mm(
            self.hypergraph_left, concept_anchor
        )
        content = self.content_projection(content)
        structural = self.structure_projection(structural)
        gate = torch.sigmoid(self.gate(torch.cat([content, structural], dim=-1)))
        return self.norm(content + gate * structural)


class ResponseConditionedRelationalRetriever(nn.Module):
    """Retrieve prior events using global logical, temporal, and response edges."""

    def __init__(self, d_model, window_size=64, dropout=0.2):
        super().__init__()
        self.window_size = int(window_size)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.edge_score = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
        self.last_attention_weights = None

    def _attend(
        self,
        memory,
        source_concepts,
        source_responses,
        target_concepts,
        target_context,
        mask,
        relation_tables,
    ):
        logic, temporal, correct, incorrect, positive, negative = relation_tables
        targets = target_concepts.unsqueeze(-1).expand_as(source_concepts)
        response = source_responses >= 0.5
        conditioned = torch.where(
            response,
            correct[source_concepts, targets],
            incorrect[source_concepts, targets],
        )
        causal = torch.where(
            response,
            positive[source_concepts, targets],
            negative[source_concepts, targets],
        )
        edge_features = torch.stack([
            logic[source_concepts, targets],
            temporal[source_concepts, targets],
            conditioned,
            causal,
        ], dim=-1)
        scores = torch.einsum(
            '...d,...wd->...w', self.query(target_context), self.key(memory)
        ) / math.sqrt(memory.size(-1))
        scores = scores + self.edge_score(edge_features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum('...w,...wd->...d', weights, self.value(memory))
        self.last_attention_weights = weights.detach()
        return self.norm(target_context + self.output(context))

    def forward_sequence(
        self,
        history,
        source_concepts,
        source_responses,
        target_concepts,
        target_context,
        seq_len,
        relation_tables,
    ):
        window = self.window_size
        padded_history = F.pad(history, (0, 0, window - 1, 0))
        memory = padded_history.unfold(1, window, 1).permute(0, 1, 3, 2)
        padded_concepts = F.pad(source_concepts, (window - 1, 0))
        concept_windows = padded_concepts.unfold(1, window, 1)
        padded_responses = F.pad(source_responses, (window - 1, 0))
        response_windows = padded_responses.unfold(1, window, 1)
        steps = history.size(1)
        positions = torch.arange(steps, device=history.device)
        offsets = torch.arange(window, device=history.device) - window + 1
        absolute = positions.unsqueeze(1) + offsets.unsqueeze(0)
        mask = absolute.unsqueeze(0) >= 0
        mask = mask & (
            absolute.unsqueeze(0) < (seq_len - 1).view(-1, 1, 1)
        )
        output = self._attend(
            memory,
            concept_windows,
            response_windows,
            target_concepts,
            target_context,
            mask,
            relation_tables,
        )
        target_valid = positions.unsqueeze(0) < (seq_len - 1).unsqueeze(1)
        return output.masked_fill(~target_valid.unsqueeze(-1), 0.0)

    def forward(
        self,
        history,
        source_concepts,
        source_responses,
        target_concepts,
        target_context,
        seq_len,
        relation_tables,
    ):
        batch_size, total_len, d_model = history.shape
        offsets = torch.arange(self.window_size, device=history.device)
        offsets = offsets - self.window_size + 1
        end = (seq_len - 1).clamp_min(0)
        positions = end.unsqueeze(1) + offsets.unsqueeze(0)
        safe = positions.clamp(min=0, max=max(total_len - 1, 0))
        batch = torch.arange(batch_size, device=history.device).unsqueeze(1)
        memory = history[batch, safe]
        concepts = source_concepts[batch, safe]
        responses = source_responses[batch, safe]
        mask = (positions >= 0) & (positions < seq_len.unsqueeze(1))
        return self._attend(
            memory,
            concepts,
            responses,
            target_concepts,
            target_context,
            mask,
            relation_tables,
        )


class CausalEvidenceStateMixer(nn.Module):
    """Preserve the sequence backbone and add graph states as gated evidence."""

    def __init__(self, d_model, dropout=0.2, n_modalities=5):
        super().__init__()
        if n_modalities not in {5, 6, 7, 8}:
            raise ValueError('causal evidence mixer expects five to eight modalities')
        self.n_modalities = int(n_modalities)
        self.sequence_router = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.sequence_bilinear = nn.Bilinear(d_model, d_model, d_model)
        self.sequence_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        evidence_count = self.n_modalities - 2
        self.evidence_router = nn.Sequential(
            nn.Linear(d_model * (evidence_count + 1), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, evidence_count),
        )
        self.cross_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.sequence_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, -2.0)

    def forward(
        self,
        h_long,
        h_short,
        graph,
        temporal,
        causal,
        target_context,
        relational=None,
        student=None,
        mastery=None,
    ):
        sequence_weights = torch.softmax(
            self.sequence_router(
                torch.cat([h_long, h_short, target_context], dim=-1)
            ),
            dim=-1,
        )
        routed_sequence = (
            sequence_weights[..., :1] * h_long
            + sequence_weights[..., 1:] * h_short
        )
        interaction = self.sequence_bilinear(h_long, h_short)
        sequence = self.sequence_norm(
            routed_sequence + self.sequence_projection(interaction)
        )

        evidence_states = [graph, temporal, causal]
        if student is not None:
            evidence_states.append(student)
        if mastery is not None:
            evidence_states.append(mastery)
        if relational is not None:
            evidence_states.append(relational)
        if len(evidence_states) != self.n_modalities - 2:
            raise ValueError('causal evidence mixer received an unexpected modality count')
        evidence_weights = torch.softmax(
            self.evidence_router(
                torch.cat([*evidence_states, target_context], dim=-1)
            ),
            dim=-1,
        )
        evidence = sum(
            evidence_weights[..., index:index + 1] * state
            for index, state in enumerate(evidence_states)
        )
        cross = self.cross_projection(sequence * evidence)
        residual_strength = torch.sigmoid(
            self.residual_gate(
                torch.cat([sequence, evidence, target_context], dim=-1)
            )
        )
        fused = self.output_norm(
            sequence + residual_strength * (evidence + cross)
        )
        diagnostics = torch.cat([
            sequence_weights,
            residual_strength * evidence_weights,
        ], dim=-1)
        return fused, diagnostics


class CausalDirectedTransitionFusion(nn.Module):
    """Add a prefix-only directed transition state without replacing V6.5."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, base, directed, target_context):
        joint = torch.cat([base, directed, target_context], dim=-1)
        update = self.update(torch.cat([
            directed,
            base * directed,
            target_context,
        ], dim=-1))
        strength = torch.sigmoid(self.gate(joint))
        return self.output_norm(base + strength * update), strength


class ReliabilityGatedMasteryResidual(nn.Module):
    """Add target-concept mastery only when a causal same-concept history exists."""

    def __init__(self, d_model, dropout=0.2, reliability_dim=4):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3 + reliability_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, base, mastery, target_context, reliability):
        update = self.update(torch.cat([
            mastery,
            target_context,
            mastery * target_context,
        ], dim=-1))
        learned_gate = torch.sigmoid(self.gate(torch.cat([
            base,
            mastery,
            target_context,
            reliability,
        ], dim=-1)))
        # reliability[..., 0] is the causal same-concept-seen indicator. This
        # makes cold-target behavior exactly equal to the V6.5 backbone.
        strength = learned_gate * reliability[..., :1]
        return base + strength * update, strength


class ResponseConditionedOutcomeGraphEncoder(nn.Module):
    """Encode a train-only source-response to target-outcome graph edge."""

    def __init__(self, edge_features, d_model, dropout=0.2):
        super().__init__()
        if edge_features.ndim != 3 or edge_features.size(-1) != 6:
            raise ValueError(
                'outcome graph features must have shape [concept, concept, 6]'
            )
        if edge_features.size(0) != edge_features.size(1):
            raise ValueError('outcome graph must be square over concepts')
        self.register_buffer('edge_features', edge_features.float())
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2 + 8, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        source_concept,
        target_concept,
        previous_response,
        source_state,
        target_context,
    ):
        edge = self.edge_features[source_concept, target_concept]
        response = previous_response.float().clamp(0.0, 1.0)
        incorrect = 1.0 - response
        selected_residual = incorrect * edge[..., 0] + response * edge[..., 1]
        opposite_residual = response * edge[..., 0] + incorrect * edge[..., 1]
        selected_confidence = incorrect * edge[..., 3] + response * edge[..., 4]
        opposite_confidence = response * edge[..., 3] + incorrect * edge[..., 4]
        signed_contrast = (2.0 * response - 1.0) * edge[..., 2]
        same_concept = (source_concept == target_concept).to(edge.dtype)
        statistics = torch.stack([
            selected_residual,
            opposite_residual,
            signed_contrast,
            selected_confidence,
            opposite_confidence,
            edge[..., 5],
            response,
            same_concept,
        ], dim=-1)
        state = self.projection(torch.cat([
            source_state,
            target_context,
            statistics,
        ], dim=-1))
        reliability = torch.maximum(
            selected_confidence, edge[..., 5]
        ).unsqueeze(-1)
        return state, reliability, statistics


class ReliabilityGatedOutcomeResidual(nn.Module):
    """Apply outcome-graph evidence only on an observed response arm."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, base, outcome, target_context, reliability):
        update = self.update(torch.cat([
            outcome,
            target_context,
            outcome * target_context,
        ], dim=-1))
        learned_gate = torch.sigmoid(self.gate(torch.cat([
            base,
            outcome,
            target_context,
            reliability,
        ], dim=-1)))
        strength = learned_gate * reliability
        return base + strength * update, strength


class GraphConditionedStateMixer(nn.Module):
    """Interaction-level MA/TRA/graph fusion followed by one latent predictor."""

    def __init__(
        self,
        d_model,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        n_modalities=5,
        joint_fusion=False,
    ):
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.joint_fusion = bool(joint_fusion)
        self.modality_embedding = nn.Parameter(
            torch.empty(self.n_modalities, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        cross_components = 3 + max(self.n_modalities - 5, 0)
        self.cross_projection = nn.Sequential(
            nn.Linear(d_model * cross_components, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.joint_projection = (
            nn.Sequential(
                nn.Linear(d_model * self.n_modalities, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            if self.joint_fusion else None
        )
        output_components = 4 if self.joint_fusion else 3
        self.output = nn.Sequential(
            nn.Linear(d_model * output_components, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.max_interaction_batch = 32768
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self,
        h_long,
        h_short,
        graph,
        temporal,
        causal,
        target_context,
        relational=None,
        student=None,
        mastery=None,
    ):
        shape = h_long.shape[:-1]
        d_model = h_long.size(-1)
        states = [h_long, h_short, graph, temporal, causal]
        if student is not None:
            states.append(student)
        if mastery is not None:
            states.append(mastery)
        if relational is not None:
            states.append(relational)
        if len(states) != self.n_modalities:
            raise ValueError('state mixer received an unexpected modality count')
        tokens = torch.stack(states, dim=-2)
        flat_tokens = tokens.reshape(-1, self.n_modalities, d_model)
        flat_tokens = flat_tokens + self.modality_embedding.unsqueeze(0)
        encoded = torch.cat([
            self.interaction_encoder(
                flat_tokens[start:start + self.max_interaction_batch]
            )
            for start in range(0, flat_tokens.size(0), self.max_interaction_batch)
        ], dim=0)
        target = target_context.reshape(-1, d_model).unsqueeze(1).expand(
            -1, self.n_modalities, -1
        )
        router_logits = self.router(torch.cat([encoded, target], dim=-1)).squeeze(-1)
        router = torch.softmax(router_logits, dim=-1)
        routed = (router.unsqueeze(-1) * encoded).sum(dim=1)

        encoded = encoded.reshape(*shape, self.n_modalities, d_model)
        cross_components = [
            encoded[..., 0, :] * encoded[..., 2, :],
            encoded[..., 1, :] * encoded[..., 4, :],
            encoded[..., 0, :] * encoded[..., 1, :],
        ]
        for index in range(5, self.n_modalities):
            cross_components.append(encoded[..., 0, :] * encoded[..., index, :])
        cross = self.cross_projection(torch.cat(cross_components, dim=-1))
        routed = routed.reshape(*shape, d_model)
        output_components = [routed, cross, target_context]
        if self.joint_projection is not None:
            output_components.append(
                self.joint_projection(encoded.flatten(start_dim=-2))
            )
        fused = self.output(torch.cat(output_components, dim=-1))
        return (
            self.output_norm(fused + routed + target_context),
            router.reshape(*shape, self.n_modalities),
        )


class MaTra4LS_KT(nn.Module):
    """V7 baseline: V6 graph-causal states prepared for fold-stable extension."""

    def __init__(
        self,
        n_questions,
        n_concepts,
        d_model=128,
        d_state=32,
        d_conv=4,
        expand=2,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        difficulty_dim=DEFAULT_DIFFICULTY_DIM,
        student_profile_dim=DEFAULT_STUDENT_PROFILE_DIM,
        task_mode='question',
        local_windows=(64,),
        mamba_version='mamba2',
        mamba_layers=2,
        short_window=None,
        summary_block_size=32,
        max_seq_len=500,
        relation_tensors=None,
        relation_names=None,
        graph_layers=2,
        state_mixer_layers=2,
        relational_history=False,
        relational_window=64,
        joint_state_fusion=False,
        state_mixer_type='transformer',
        causal_prefix_fusion=False,
        student_incidence=None,
        outcome_graph_features=None,
        causal_directed_graph=False,
        target_mastery_state=False,
        target_mastery_fusion='evidence',
        mastery_auxiliary_head=False,
        prior_auxiliary_head=False,
        trainable_student_anchor=False,
        auxiliary_heads=False,
        branch_ensemble_prediction=False,
        structured_profile_residual=False,
        mastery_history_size=8,
        **unused,
    ):
        super().__init__()
        del unused
        self.task_mode = str(task_mode).strip().lower()
        if self.task_mode not in {'concept', 'question'}:
            raise ValueError("task_mode must be either 'concept' or 'question'")
        self.n_questions = int(n_questions)
        self.n_concepts = int(n_concepts)
        self.d_model = int(d_model)
        self.difficulty_dim = int(difficulty_dim)
        self.student_profile_dim = int(student_profile_dim)
        self.structured_profile_residual = bool(structured_profile_residual)
        self.base_profile_dim = (
            min(self.student_profile_dim, 13)
            if self.structured_profile_residual else self.student_profile_dim
        )
        self.structured_profile_dim = (
            max(self.student_profile_dim - 13, 0)
            if self.structured_profile_residual else 0
        )
        if self.task_mode == 'question' and self.n_questions < 1:
            raise ValueError('question-level mode requires real question IDs')
        if relation_tensors is None:
            relation_tensors = torch.eye(self.n_concepts).unsqueeze(0)
            relation_names = ('identity',)
        if relation_tensors.size(1) != self.n_concepts:
            raise ValueError('relation tensor vocabulary does not match n_concepts')
        self.relation_names = tuple(
            relation_names or [f'relation_{i}' for i in range(relation_tensors.size(0))]
        )
        if len(self.relation_names) != relation_tensors.size(0):
            raise ValueError('relation_names and relation_tensors must have equal length')
        self.relation_index = {
            name: index for index, name in enumerate(self.relation_names)
        }
        self.relational_history = bool(relational_history)
        self.state_mixer_type = str(state_mixer_type).strip().lower()
        self.causal_prefix_fusion = bool(causal_prefix_fusion)
        self.causal_directed_graph = bool(causal_directed_graph)
        self.target_mastery_state = bool(target_mastery_state)
        self.mastery_auxiliary_head = bool(mastery_auxiliary_head)
        self.prior_auxiliary_head = bool(prior_auxiliary_head)
        self.auxiliary_heads = bool(auxiliary_heads)
        self.branch_ensemble_prediction = bool(branch_ensemble_prediction)
        if self.branch_ensemble_prediction and not self.auxiliary_heads:
            raise ValueError(
                'branch_ensemble_prediction requires auxiliary_heads'
            )
        if self.mastery_auxiliary_head and not self.target_mastery_state:
            raise ValueError(
                'mastery_auxiliary_head requires target_mastery_state'
            )
        self.target_mastery_fusion = str(target_mastery_fusion).strip().lower()
        if self.target_mastery_fusion not in {'evidence', 'residual'}:
            raise ValueError(
                "target_mastery_fusion must be either 'evidence' or 'residual'"
            )
        if self.state_mixer_type not in {'transformer', 'causal_evidence'}:
            raise ValueError(
                "state_mixer_type must be 'transformer' or 'causal_evidence'"
            )

        if short_window is None:
            values = list(local_windows) if local_windows is not None else [64]
            short_window = max(int(value) for value in values)
        self.concept_emb = nn.Embedding(self.n_concepts, d_model)
        self.question_emb = (
            nn.Embedding(self.n_questions, d_model)
            if self.task_mode == 'question' else None
        )
        self.response_emb = nn.Embedding(2, d_model)
        self.interaction_emb = nn.Embedding(self.n_concepts * 2, d_model)
        self.time_encoder = TimeEncoding(d_model)
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(self.difficulty_dim, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(self.base_profile_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.structured_profile_encoder = (
            nn.Sequential(
                nn.Linear(self.structured_profile_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.structured_profile_dim else None
        )
        self.structured_profile_strength = (
            nn.Parameter(torch.tensor(-4.0))
            if self.structured_profile_dim else None
        )
        self.graph_encoder = MultiRelationConceptEncoder(
            relation_tensors, d_model, n_layers=graph_layers, dropout=dropout
        )
        self.student_graph_encoder = (
            StudentConceptBipartiteEncoder(
                student_incidence,
                d_model=d_model,
                dropout=dropout,
                trainable_anchor=trainable_student_anchor,
            )
            if student_incidence is not None else None
        )
        self.outcome_graph_encoder = (
            ResponseConditionedOutcomeGraphEncoder(
                outcome_graph_features,
                d_model=d_model,
                dropout=dropout,
            )
            if outcome_graph_features is not None else None
        )

        student_component = 1 if self.student_graph_encoder is not None else 0
        event_components = (
            9 + student_component + (1 if self.task_mode == 'question' else 0)
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_components * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.long_event_adapter = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.short_event_adapter = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.transition_start = (
            nn.Parameter(torch.zeros(d_model))
            if self.causal_directed_graph else None
        )
        self.directed_event_projection = (
            nn.Sequential(
                nn.Linear(d_model * 7, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
            if self.causal_directed_graph else None
        )
        target_components = (
            6 + student_component + (1 if self.task_mode == 'question' else 0)
        )
        self.target_projection = nn.Sequential(
            nn.Linear(target_components * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.long_term = TargetConditionedLongMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=mamba_layers,
            dropout=dropout,
            version=mamba_version,
        )
        self.short_term = TargetCrossAttentionTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            window_size=int(short_window),
            summary_block_size=summary_block_size,
            max_seq_len=max_seq_len,
        )
        self.directed_transition_encoder = (
            TargetConditionedLongMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_layers=mamba_layers,
                dropout=dropout,
                version=mamba_version,
            )
            if self.causal_directed_graph else None
        )
        self.target_mastery_tracker = (
            DualTimescaleConceptStateTracker(
                d_model=d_model,
                dropout=dropout,
                max_history=mastery_history_size,
            )
            if self.target_mastery_state else None
        )
        self.relational_retriever = (
            ResponseConditionedRelationalRetriever(
                d_model=d_model,
                window_size=relational_window,
                dropout=dropout,
            )
            if self.relational_history else None
        )
        modality_count = (
            5
            + int(self.relational_history)
            + int(self.student_graph_encoder is not None)
            + int(
                self.target_mastery_state
                and self.target_mastery_fusion == 'evidence'
            )
        )
        if self.state_mixer_type == 'causal_evidence':
            self.state_mixer = CausalEvidenceStateMixer(
                d_model=d_model,
                dropout=dropout,
                n_modalities=modality_count,
            )
        else:
            self.state_mixer = GraphConditionedStateMixer(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=state_mixer_layers,
                dropout=dropout,
                n_modalities=modality_count,
                joint_fusion=joint_state_fusion,
            )
        self.directed_transition_fusion = (
            CausalDirectedTransitionFusion(d_model=d_model, dropout=dropout)
            if self.causal_directed_graph else None
        )
        self.mastery_residual_fusion = (
            ReliabilityGatedMasteryResidual(d_model=d_model, dropout=dropout)
            if self.target_mastery_state
            and self.target_mastery_fusion == 'residual'
            else None
        )
        self.outcome_graph_residual_fusion = (
            ReliabilityGatedOutcomeResidual(d_model=d_model, dropout=dropout)
            if self.outcome_graph_encoder is not None else None
        )
        self.mastery_predictor = (
            nn.Sequential(
                nn.Linear(d_model * 2 + 4, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.mastery_auxiliary_head else None
        )
        self.prior_predictor = (
            nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.prior_auxiliary_head else None
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.long_predictor = (
            nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.auxiliary_heads else None
        )
        self.short_predictor = (
            nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.auxiliary_heads else None
        )
        self.concept_bias = nn.Embedding(self.n_concepts, 1)
        self.causal_prior_gate = (
            nn.Sequential(
                nn.Linear(d_model * 2 + 5, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.causal_prefix_fusion else None
        )
        self.apply(self._init_weights)
        if self.structured_profile_encoder is not None:
            nn.init.zeros_(self.structured_profile_encoder[-1].weight)
            nn.init.zeros_(self.structured_profile_encoder[-1].bias)
        if isinstance(self.state_mixer, CausalEvidenceStateMixer):
            nn.init.zeros_(self.state_mixer.residual_gate[-1].weight)
            nn.init.constant_(self.state_mixer.residual_gate[-1].bias, -2.0)
        if self.causal_prior_gate is not None:
            nn.init.zeros_(self.causal_prior_gate[-1].weight)
            nn.init.zeros_(self.causal_prior_gate[-1].bias)
        if self.directed_transition_fusion is not None:
            nn.init.normal_(
                self.directed_transition_fusion.update[-1].weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.directed_transition_fusion.update[-1].bias)
            nn.init.zeros_(self.directed_transition_fusion.gate[-1].weight)
            nn.init.constant_(self.directed_transition_fusion.gate[-1].bias, -2.0)
        if self.mastery_residual_fusion is not None:
            nn.init.normal_(
                self.mastery_residual_fusion.update[-1].weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.mastery_residual_fusion.update[-1].bias)
            nn.init.zeros_(self.mastery_residual_fusion.gate[-1].weight)
            nn.init.constant_(self.mastery_residual_fusion.gate[-1].bias, -1.0)
        if self.outcome_graph_residual_fusion is not None:
            nn.init.normal_(
                self.outcome_graph_residual_fusion.update[-1].weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.outcome_graph_residual_fusion.update[-1].bias)
            nn.init.zeros_(self.outcome_graph_residual_fusion.gate[-1].weight)
            nn.init.constant_(
                self.outcome_graph_residual_fusion.gate[-1].bias, -1.0
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _match_feature_dim(feature, target_dim):
        if feature.size(-1) > target_dim:
            return feature[..., :target_dim].float()
        if feature.size(-1) < target_dim:
            return F.pad(feature.float(), (0, target_dim - feature.size(-1)))
        return feature.float()

    def _feature_or_zeros(self, feature, shape, target_dim, device):
        if feature is None:
            return torch.zeros(*shape, target_dim, device=device)
        return self._match_feature_dim(feature.to(device), target_dim)

    def _embed_profile(self, profile):
        profile = self._match_feature_dim(profile, self.student_profile_dim)
        embedded = self.profile_encoder(profile[..., :self.base_profile_dim])
        if self.structured_profile_encoder is not None:
            structured = self.structured_profile_encoder(
                profile[..., self.base_profile_dim:]
            )
            embedded = embedded + torch.sigmoid(
                self.structured_profile_strength
            ) * structured
        return embedded

    def _relation(self, relation_states, name, fallback):
        index = self.relation_index.get(name)
        return fallback if index is None else relation_states[index]

    def _concept_states(self):
        return self.graph_encoder(self.concept_emb.weight)

    def _student_state(self, batch, graph_table):
        if self.student_graph_encoder is None:
            return None
        if batch.get('student_id') is None:
            raise ValueError('student bipartite graph requires batch.student_id')
        table = self.student_graph_encoder(graph_table)
        student_id = batch['student_id'].long()
        if torch.any(student_id < 0) or torch.any(student_id >= table.size(0)):
            raise ValueError('batch contains a student outside the incidence graph')
        return table[student_id]

    @staticmethod
    def _expand_student_state(student_state, target_context):
        if student_state is None:
            return None
        expanded = student_state
        while expanded.ndim < target_context.ndim:
            expanded = expanded.unsqueeze(-2)
        return expanded.expand_as(target_context)

    def _relational_tables(self):
        fallback = getattr(self.graph_encoder, 'relation_0').to_dense()
        names = (
            'student_logic',
            'temporal_forward',
            'previous_correct',
            'previous_incorrect',
            'causal_positive',
            'causal_negative',
        )
        output = []
        for name in names:
            index = self.relation_index.get(name)
            output.append(
                fallback if index is None else
                getattr(self.graph_encoder, f'relation_{index}').to_dense()
            )
        return tuple(output)

    def _event_states(
        self,
        question_seq,
        concept_seq,
        response_seq,
        time_gap_seq,
        difficulty_seq,
        profile_seq,
        graph_table,
        relation_states,
        student_state=None,
    ):
        response = response_seq.long().clamp(min=0, max=1)
        base = self.concept_emb(concept_seq)
        graph = graph_table[concept_seq]
        correct_table = self._relation(relation_states, 'previous_correct', graph_table)
        incorrect_table = self._relation(relation_states, 'previous_incorrect', graph_table)
        positive_table = self._relation(relation_states, 'causal_positive', graph_table)
        negative_table = self._relation(relation_states, 'causal_negative', graph_table)
        conditioned = torch.where(
            response.unsqueeze(-1).bool(),
            correct_table[concept_seq],
            incorrect_table[concept_seq],
        )
        causal = torch.where(
            response.unsqueeze(-1).bool(),
            positive_table[concept_seq],
            negative_table[concept_seq],
        )
        components = [
            base,
            graph,
            conditioned,
            causal,
            self.response_emb(response),
            self.interaction_emb(concept_seq + response * self.n_concepts),
            self.time_encoder(time_gap_seq.float()),
            self.difficulty_encoder(difficulty_seq),
            self._embed_profile(profile_seq),
        ]
        if student_state is not None:
            components.append(
                student_state.unsqueeze(1).expand(-1, concept_seq.size(1), -1)
            )
        if self.task_mode == 'question':
            if question_seq is None:
                raise ValueError('question-level mode requires question_seq')
            components.append(self.question_emb(question_seq))
        event = self.event_projection(torch.cat(components, dim=-1))
        directed_event = None
        if self.directed_event_projection is not None:
            previous_graph = torch.cat([
                self.transition_start.view(1, 1, -1).expand(
                    graph.size(0), 1, -1
                ),
                graph[:, :-1],
            ], dim=1)
            directed_event = self.directed_event_projection(torch.cat([
                previous_graph,
                graph,
                graph - previous_graph,
                graph * previous_graph,
                conditioned,
                causal,
                self.response_emb(response),
            ], dim=-1))
        return (
            self.long_event_adapter(event),
            self.short_event_adapter(event),
            directed_event,
        )

    def _fuse_directed_transition(self, fused, directed, target_context):
        if self.directed_transition_fusion is None:
            return fused, None
        return self.directed_transition_fusion(fused, directed, target_context)

    def _fuse_target_mastery(
        self,
        fused,
        mastery,
        target_context,
        target_profile,
    ):
        if self.mastery_residual_fusion is None:
            return fused, None
        reliability = self._target_mastery_reliability(target_profile)
        return self.mastery_residual_fusion(
            fused,
            mastery,
            target_context,
            reliability,
        )

    def _outcome_graph_state(
        self,
        source_concept,
        target_concept,
        previous_response,
        source_state,
        target_context,
    ):
        if self.outcome_graph_encoder is None:
            return None, None, None
        return self.outcome_graph_encoder(
            source_concept,
            target_concept,
            previous_response,
            source_state,
            target_context,
        )

    def _fuse_outcome_graph(
        self,
        fused,
        outcome,
        target_context,
        reliability,
    ):
        if self.outcome_graph_residual_fusion is None:
            return fused, None
        return self.outcome_graph_residual_fusion(
            fused,
            outcome,
            target_context,
            reliability,
        )

    def _target_mastery_reliability(self, target_profile):
        seen = self._profile_value(
            target_profile,
            8,
            target_profile.new_zeros(target_profile.shape[:-1]),
        )
        attempt_signal = self._profile_value(
            target_profile,
            6,
            target_profile.new_zeros(target_profile.shape[:-1]),
        )
        confidence = self._profile_value(target_profile, 16, attempt_signal)
        lag = self._profile_value(
            target_profile,
            9,
            target_profile.new_ones(target_profile.shape[:-1]),
        )
        return torch.stack([
            seen.clamp(0.0, 1.0),
            attempt_signal.clamp(0.0, 1.0),
            confidence.clamp(0.0, 1.0),
            (1.0 - lag).clamp(0.0, 1.0),
        ], dim=-1)

    def _mastery_logits(
        self,
        mastery,
        target_context,
        target_profile,
        target_concept,
    ):
        if self.mastery_predictor is None:
            return None
        reliability = self._target_mastery_reliability(target_profile)
        logits = self.mastery_predictor(torch.cat([
            mastery,
            target_context,
            reliability,
        ], dim=-1)).squeeze(-1)
        return logits + self.concept_bias(target_concept).squeeze(-1)

    def _structural_prior_logits(
        self,
        target_context,
        graph,
        causal,
        target_concept,
    ):
        if self.prior_predictor is None:
            return None
        logits = self.prior_predictor(torch.cat([
            target_context,
            graph,
            causal,
        ], dim=-1)).squeeze(-1)
        return logits + self.concept_bias(target_concept).squeeze(-1)

    def _mastery_state(self, events, target_context, batch, target_positions):
        if self.target_mastery_tracker is None:
            return None
        indices = batch.get('mastery_history_indices')
        if indices is None:
            raise ValueError(
                'target mastery state requires batch.mastery_history_indices'
            )
        return self.target_mastery_tracker(
            events,
            target_context,
            indices,
            target_positions,
        )

    def _target_states(
        self,
        target_question,
        target_concept,
        target_difficulty,
        target_profile,
        graph_table,
        relation_states,
        student_state=None,
    ):
        graph = graph_table[target_concept]
        temporal = 0.5 * (
            self._relation(relation_states, 'temporal_forward', graph_table)[target_concept]
            + self._relation(relation_states, 'temporal_reverse', graph_table)[target_concept]
        )
        causal = (
            self._relation(relation_states, 'causal_positive', graph_table)[target_concept]
            - self._relation(relation_states, 'causal_negative', graph_table)[target_concept]
        )
        components = [
            self.concept_emb(target_concept),
            graph,
            temporal,
            causal,
            self.difficulty_encoder(target_difficulty),
            self._embed_profile(target_profile),
        ]
        if student_state is not None:
            expanded_student = student_state
            while expanded_student.ndim < target_concept.ndim + 1:
                expanded_student = expanded_student.unsqueeze(1)
            components.append(expanded_student.expand(*target_concept.shape, -1))
        if self.task_mode == 'question':
            if target_question is None:
                raise ValueError('question-level mode requires target question IDs')
            components.append(self.question_emb(target_question))
        target_context = self.target_projection(torch.cat(components, dim=-1))
        return target_context, graph, temporal, causal

    def _logits(
        self,
        fused,
        target_context,
        graph,
        causal,
        target_concept,
    ):
        logits = self.predictor(
            torch.cat([fused, target_context, graph, causal], dim=-1)
        ).squeeze(-1)
        logits = logits + self.concept_bias(target_concept).squeeze(-1)
        return logits

    def _branch_logits(
        self,
        h_long,
        h_short,
        target_context,
        graph,
        causal,
        target_concept,
    ):
        if self.long_predictor is None or self.short_predictor is None:
            return None, None
        concept_bias = self.concept_bias(target_concept).squeeze(-1)
        long_logits = self.long_predictor(
            torch.cat([h_long, target_context, graph, causal], dim=-1)
        ).squeeze(-1) + concept_bias
        short_logits = self.short_predictor(
            torch.cat([h_short, target_context, graph, causal], dim=-1)
        ).squeeze(-1) + concept_bias
        return long_logits, short_logits

    @staticmethod
    def _average_probability_logits(*logits):
        probability = torch.stack(
            [torch.sigmoid(value) for value in logits], dim=0
        ).mean(dim=0)
        return torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))

    @staticmethod
    def _profile_value(profile, index, fallback):
        if profile.size(-1) <= index:
            return fallback
        return profile[..., index]

    def _causal_prefix_prior(self, target_difficulty, target_profile):
        eps = 1e-4
        overall = self._profile_value(
            target_profile, 0, target_profile.new_full(target_profile.shape[:-1], 0.5)
        )
        concept_accuracy = self._profile_value(target_profile, 5, overall)
        attempt_signal = self._profile_value(
            target_profile, 6, target_profile.new_zeros(target_profile.shape[:-1])
        )
        seen = self._profile_value(
            target_profile, 8, (attempt_signal > 0).to(target_profile.dtype)
        )
        concept_lag = self._profile_value(
            target_profile, 9, target_profile.new_ones(target_profile.shape[:-1])
        )
        recent = self._profile_value(target_profile, 11, concept_accuracy)
        ema = self._profile_value(target_profile, 12, recent)
        bayesian = self._profile_value(target_profile, 15, concept_accuracy)
        confidence = self._profile_value(target_profile, 16, attempt_signal)
        confidence = confidence.clamp(0.0, 1.0) * seen.clamp(0.0, 1.0)

        if target_difficulty.size(-1) > 1:
            concept_base = 1.0 - target_difficulty[..., 1]
        else:
            concept_base = overall
        cold_probability = 0.7 * concept_base + 0.3 * overall
        repeat_probability = (
            0.35 * concept_accuracy
            + 0.25 * recent
            + 0.25 * ema
            + 0.15 * bayesian
        )
        prior_probability = (
            cold_probability + confidence * (repeat_probability - cold_probability)
        ).clamp(eps, 1.0 - eps)
        prior_logit = torch.logit(prior_probability)
        evidence = torch.stack([
            seen.clamp(0.0, 1.0),
            confidence,
            attempt_signal.clamp(0.0, 1.0),
            concept_lag.clamp(0.0, 1.0),
            (repeat_probability - cold_probability).abs().clamp(0.0, 1.0),
        ], dim=-1)
        return prior_logit, evidence

    def _fuse_causal_prefix_prior(
        self,
        logits,
        fused,
        target_context,
        target_difficulty,
        target_profile,
    ):
        if self.causal_prior_gate is None:
            return logits, None, None
        prior_logits, evidence = self._causal_prefix_prior(
            target_difficulty, target_profile
        )
        learned_gate = torch.sigmoid(
            self.causal_prior_gate(
                torch.cat([fused, target_context, evidence], dim=-1)
            ).squeeze(-1)
        )
        gate = learned_gate * evidence[..., 1]
        return logits + gate * (prior_logits - logits), prior_logits, gate

    def forward(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        concept_seq = batch['concept_seq']
        response_seq = batch['response_seq']
        seq_len = batch['seq_len']
        batch_size, total_len = concept_seq.shape
        difficulty = self._feature_or_zeros(
            batch.get('difficulty_seq'), (batch_size, total_len),
            self.difficulty_dim, concept_seq.device
        )
        profile = self._feature_or_zeros(
            batch.get('student_profile_seq'), (batch_size, total_len),
            self.student_profile_dim, concept_seq.device
        )
        graph_table, relation_states, relation_gate = self._concept_states()
        student_state = self._student_state(batch, graph_table)
        long_x, short_x, directed_x = self._event_states(
            batch.get('question_seq'), concept_seq, response_seq,
            batch['time_gap_seq'], difficulty, profile, graph_table, relation_states,
            student_state,
        )
        target_difficulty = self._feature_or_zeros(
            batch.get('target_difficulty'), (batch_size,),
            self.difficulty_dim, concept_seq.device
        )
        target_profile = self._feature_or_zeros(
            batch.get('target_profile'), (batch_size,),
            self.student_profile_dim, concept_seq.device
        )
        target_context, graph, temporal, causal = self._target_states(
            batch.get('target_question'), batch['target_concept'],
            target_difficulty, target_profile,
            graph_table, relation_states, student_state
        )
        mastery = self._mastery_state(
            short_x,
            target_context,
            batch,
            seq_len,
        )
        h_long = self.long_term(long_x, seq_len, target_context)
        h_short = self.short_term(short_x, seq_len, target_context)
        h_directed = (
            self.directed_transition_encoder(directed_x, seq_len, target_context)
            if self.directed_transition_encoder is not None else None
        )
        relational = (
            self.relational_retriever(
                short_x,
                concept_seq,
                response_seq,
                batch['target_concept'],
                target_context,
                seq_len,
                self._relational_tables(),
            )
            if self.relational_retriever is not None else None
        )
        source_position = (seq_len - 1).clamp(min=0, max=total_len - 1)
        batch_index = torch.arange(batch_size, device=concept_seq.device)
        outcome, outcome_reliability, outcome_statistics = (
            self._outcome_graph_state(
                concept_seq[batch_index, source_position],
                batch['target_concept'],
                response_seq[batch_index, source_position],
                short_x[batch_index, source_position],
                target_context,
            )
        )
        fused, state_gate = self.state_mixer(
            h_long,
            h_short,
            graph,
            temporal,
            causal,
            target_context,
            relational,
            self._expand_student_state(student_state, target_context),
            (
                mastery
                if self.target_mastery_fusion == 'evidence'
                else None
            ),
        )
        fused, mastery_residual_gate = self._fuse_target_mastery(
            fused,
            mastery,
            target_context,
            target_profile,
        )
        fused, outcome_graph_gate = self._fuse_outcome_graph(
            fused,
            outcome,
            target_context,
            outcome_reliability,
        )
        fused, directed_transition_gate = self._fuse_directed_transition(
            fused, h_directed, target_context
        )
        logits = self._logits(
            fused, target_context, graph, causal, batch['target_concept']
        )
        mastery_logits = self._mastery_logits(
            mastery,
            target_context,
            target_profile,
            batch['target_concept'],
        )
        prior_logits = self._structural_prior_logits(
            target_context,
            graph,
            causal,
            batch['target_concept'],
        )
        logits, causal_prior_logits, causal_prior_gate = self._fuse_causal_prefix_prior(
            logits,
            fused,
            target_context,
            target_difficulty,
            target_profile,
        )
        main_logits = logits
        long_logits, short_logits = self._branch_logits(
            h_long,
            h_short,
            target_context,
            graph,
            causal,
            batch['target_concept'],
        )
        if self.branch_ensemble_prediction:
            logits = self._average_probability_logits(
                main_logits, long_logits, short_logits
            )
        if not return_aux:
            return torch.sigmoid(logits)
        output = {
            'logits': logits,
            'state_gate': state_gate,
            'relation_gate': relation_gate,
            'attention_weights': self.short_term.last_attention_weights,
            'relational_attention_weights': (
                self.relational_retriever.last_attention_weights
                if self.relational_retriever is not None else None
            ),
            'mastery_timescale_weights': (
                self.target_mastery_tracker.last_attention_weights
                if self.target_mastery_tracker is not None else None
            ),
        }
        if long_logits is not None:
            output['main_logits'] = main_logits
            output['long_logits'] = long_logits
            output['short_logits'] = short_logits
        if directed_transition_gate is not None:
            output['directed_transition_gate'] = directed_transition_gate
        if mastery_residual_gate is not None:
            output['mastery_residual_gate'] = mastery_residual_gate
        if outcome_graph_gate is not None:
            output['outcome_graph_gate'] = outcome_graph_gate
            output['outcome_graph_statistics'] = outcome_statistics
        if mastery_logits is not None:
            output['mastery_logits'] = mastery_logits
        if prior_logits is not None:
            output['prior_logits'] = prior_logits
        if causal_prior_logits is not None:
            output['causal_prior_logits'] = causal_prior_logits
            output['causal_prior_gate'] = causal_prior_gate
        return output

    def forward_sequence(self, batch, relation_matrix=None, return_aux=False):
        del relation_matrix
        concept_seq = batch['concept_seq']
        response_seq = batch['response_seq']
        seq_len = batch['seq_len']
        batch_size, total_len = concept_seq.shape
        if total_len < 2:
            empty = concept_seq.new_zeros(batch_size, 0, dtype=torch.float)
            return {'logits': empty} if return_aux else empty
        difficulty = self._feature_or_zeros(
            batch.get('difficulty_seq'), (batch_size, total_len),
            self.difficulty_dim, concept_seq.device
        )
        profile = self._feature_or_zeros(
            batch.get('student_profile_seq'), (batch_size, total_len),
            self.student_profile_dim, concept_seq.device
        )
        graph_table, relation_states, relation_gate = self._concept_states()
        student_state = self._student_state(batch, graph_table)
        long_x, short_x, directed_x = self._event_states(
            batch.get('question_seq'), concept_seq, response_seq,
            batch['time_gap_seq'], difficulty, profile, graph_table, relation_states,
            student_state,
        )
        target_question = (
            batch['question_seq'][:, 1:] if batch.get('question_seq') is not None else None
        )
        target_concept = concept_seq[:, 1:]
        target_context, graph, temporal, causal = self._target_states(
            target_question, target_concept, difficulty[:, 1:], profile[:, 1:],
            graph_table, relation_states, student_state
        )
        target_positions = torch.arange(
            1, total_len, device=concept_seq.device
        ).unsqueeze(0).expand(batch_size, -1)
        mastery = self._mastery_state(
            short_x,
            target_context,
            batch,
            target_positions,
        )
        h_long = self.long_term.forward_sequence(long_x, seq_len, target_context)
        h_short = self.short_term.forward_sequence(short_x, seq_len, target_context)
        h_directed = (
            self.directed_transition_encoder.forward_sequence(
                directed_x, seq_len, target_context
            )
            if self.directed_transition_encoder is not None else None
        )
        relational = (
            self.relational_retriever.forward_sequence(
                short_x[:, :-1],
                concept_seq[:, :-1],
                response_seq[:, :-1],
                target_concept,
                target_context,
                seq_len,
                self._relational_tables(),
            )
            if self.relational_retriever is not None else None
        )
        outcome, outcome_reliability, outcome_statistics = (
            self._outcome_graph_state(
                concept_seq[:, :-1],
                target_concept,
                response_seq[:, :-1],
                short_x[:, :-1],
                target_context,
            )
        )
        fused, state_gate = self.state_mixer(
            h_long,
            h_short,
            graph,
            temporal,
            causal,
            target_context,
            relational,
            self._expand_student_state(student_state, target_context),
            (
                mastery
                if self.target_mastery_fusion == 'evidence'
                else None
            ),
        )
        fused, mastery_residual_gate = self._fuse_target_mastery(
            fused,
            mastery,
            target_context,
            profile[:, 1:],
        )
        fused, outcome_graph_gate = self._fuse_outcome_graph(
            fused,
            outcome,
            target_context,
            outcome_reliability,
        )
        fused, directed_transition_gate = self._fuse_directed_transition(
            fused, h_directed, target_context
        )
        logits = self._logits(
            fused,
            target_context,
            graph,
            causal,
            target_concept,
        )
        mastery_logits = self._mastery_logits(
            mastery,
            target_context,
            profile[:, 1:],
            target_concept,
        )
        prior_logits = self._structural_prior_logits(
            target_context,
            graph,
            causal,
            target_concept,
        )
        logits, causal_prior_logits, causal_prior_gate = self._fuse_causal_prefix_prior(
            logits,
            fused,
            target_context,
            difficulty[:, 1:],
            profile[:, 1:],
        )
        main_logits = logits
        long_logits, short_logits = self._branch_logits(
            h_long,
            h_short,
            target_context,
            graph,
            causal,
            target_concept,
        )
        if self.branch_ensemble_prediction:
            logits = self._average_probability_logits(
                main_logits, long_logits, short_logits
            )
        positions = torch.arange(total_len - 1, device=concept_seq.device).unsqueeze(0)
        valid = positions < (seq_len - 1).unsqueeze(1)
        logits = logits.masked_fill(~valid, 0.0)
        main_logits = main_logits.masked_fill(~valid, 0.0)
        if not return_aux:
            return torch.sigmoid(logits)
        output = {
            'logits': logits,
            'state_gate': state_gate,
            'relation_gate': relation_gate,
            'attention_weights': self.short_term.last_attention_weights,
            'relational_attention_weights': (
                self.relational_retriever.last_attention_weights
                if self.relational_retriever is not None else None
            ),
            'mastery_timescale_weights': (
                self.target_mastery_tracker.last_attention_weights
                if self.target_mastery_tracker is not None else None
            ),
        }
        if long_logits is not None:
            output['main_logits'] = main_logits
            output['long_logits'] = long_logits.masked_fill(~valid, 0.0)
            output['short_logits'] = short_logits.masked_fill(~valid, 0.0)
        if directed_transition_gate is not None:
            output['directed_transition_gate'] = directed_transition_gate
        if mastery_residual_gate is not None:
            output['mastery_residual_gate'] = mastery_residual_gate
        if outcome_graph_gate is not None:
            output['outcome_graph_gate'] = outcome_graph_gate
            output['outcome_graph_statistics'] = outcome_statistics
        if mastery_logits is not None:
            output['mastery_logits'] = mastery_logits.masked_fill(~valid, 0.0)
        if prior_logits is not None:
            output['prior_logits'] = prior_logits.masked_fill(~valid, 0.0)
        if causal_prior_logits is not None:
            output['causal_prior_logits'] = causal_prior_logits
            output['causal_prior_gate'] = causal_prior_gate
        return output
