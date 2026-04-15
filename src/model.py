import torch
from transformers import AutoModelForCausalLM
import paged_attn
from cache_manager import KVCacheManager

class PagedAttentionModel(torch.nn.Module):
    """Wrapper around real HF weights to execute custom CUDA paged attention."""

    def __init__(self, model_id: str, device="cuda"):
        super().__init__()
        self.device = device

        print(f"Loading weights from {model_id}...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)

        self.embed = hf_model.model.embed_tokens
        self.rotary_emb = hf_model.model.rotary_emb
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

        self.num_heads = hf_model.config.num_attention_heads
        self.num_kv_heads = hf_model.config.num_key_value_heads
        self.head_dim = hf_model.config.hidden_size // self.num_heads

    @torch.no_grad()
    def forward_layer(
        self, x, layer, k_cache, v_cache,
        block_table_tensor, cache_seqlens, block_size,
        position_ids,
    ):
        batch_size = x.size(0)
        normed = layer.input_layernorm(x)

        q = layer.self_attn.q_proj(normed).view(batch_size, 1, self.num_heads, self.head_dim)
        k = layer.self_attn.k_proj(normed).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v = layer.self_attn.v_proj(normed).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        # Apply RoPE: get cos/sin from the shared model-level rotary_emb,
        # then apply via the per-layer rotary_fn (which IS apply_rotary_pos_emb)
        q_rot = q.transpose(1, 2)  # [B, num_heads, 1, head_dim]
        k_rot = k.transpose(1, 2)  # [B, num_kv_heads, 1, head_dim]
        cos, sin = self.rotary_emb(v.transpose(1, 2), position_ids)
        q_rot, k_rot = layer.self_attn.rotary_fn(q_rot, k_rot, cos, sin)
        q = q_rot.transpose(1, 2)  # [B, 1, num_heads, head_dim]
        k = k_rot.transpose(1, 2)  # [B, 1, num_kv_heads, head_dim]

        # Write current token's KV into the paged cache
        paged_attn.update(
            k_cache, v_cache,
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            block_table_tensor,
            block_size,
            cache_seqlens,
        )

        # Flatten q back to [B, 1, num_heads * head_dim] for the decode kernel
        attn_out = paged_attn.decode(
            q.reshape(batch_size, 1, self.num_heads * self.head_dim).contiguous(),
            k_cache, v_cache,
            block_table_tensor,
            block_size,
            cache_seqlens + 1,
            self.num_heads,
            True,
        )

        x = x + layer.self_attn.o_proj(attn_out)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x

    @torch.no_grad()
    def forward_batch(self, batch_tokens: list[int], batch_positions: list[int], batch_block_tables: list[list[int]], kv_manager) -> list[int]:
        batch_size = len(batch_tokens)

        # 1. Embed tokens: [batch_size, 1, hidden_dim]
        x = self.embed(torch.tensor(batch_tokens, device=self.device).unsqueeze(1))

        # 2. Pad block tables to a square tensor
        max_blocks = max(len(t) for t in batch_block_tables)
        padded_tables = [t + [0] * (max_blocks - len(t)) for t in batch_block_tables]
        block_table_tensor = torch.tensor(padded_tables, dtype=torch.int32, device=self.device)

        # cache_seqlens[i] = number of tokens already in cache for request i (= current pos)
        cache_seqlens = torch.tensor(batch_positions, dtype=torch.int32, device=self.device)

        # position_ids[i] = the absolute position of the token being processed now
        position_ids = cache_seqlens.long().unsqueeze(1)  # [B, 1]

        # 3. Run through all transformer layers with RoPE-aware paged attention
        for layer_idx, layer in enumerate(self.layers):
            k_cache, v_cache = kv_manager.get_layer_cache(layer_idx)
            x = self.forward_layer(
                x, layer, k_cache, v_cache,
                block_table_tensor, cache_seqlens, kv_manager.block_size,
                position_ids,
            )

        # 4. Final norm → LM head → greedy argmax
        logits = self.lm_head(self.norm(x))
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        return next_tokens
