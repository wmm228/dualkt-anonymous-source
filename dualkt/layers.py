"""Mamba-2 and hierarchical target-attention layers used by DualKT."""

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


