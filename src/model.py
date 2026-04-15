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
        self.layer = hf_model.model.layers[0]
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

        self.num_heads = hf_model.config.num_attention_heads
        self.num_kv_heads = hf_model.config.num_key_value_heads
        self.head_dim = hf_model.config.hidden_size // self.num_heads

    @torch.no_grad()
    def forward_layer(
        self, x, layer, k_cache, v_cache,
        block_table_tensor, cache_seqlens, block_size
    ):
        batch_size = x.size(0)
        normed = layer.input_layernorm(x)

        q = layer.self_attn.q_proj(normed)          # [B, 1, num_heads * head_dim]
        k = layer.self_attn.k_proj(normed).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v = layer.self_attn.v_proj(normed).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        paged_attn.update(
            k_cache, v_cache,
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            block_table_tensor,
            block_size,
            cache_seqlens,
        )

        attn_out = paged_attn.decode(
            q.contiguous(),
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
        
        # 1. Embed the whole batch at once: [batch_size, 1, head_dim]
        x = self.embed(torch.tensor(batch_tokens, device=self.device).unsqueeze(1))

        # 3. Pad block tables so they can be converted to a square tensor
        max_blocks = max(len(t) for t in batch_block_tables)
        padded_tables = [t + [0]*(max_blocks - len(t)) for t in batch_block_tables]

        block_table_tensor = torch.tensor(padded_tables, dtype=torch.int32, device=self.device)
        cache_seqlens = torch.tensor(batch_positions, dtype=torch.int32, device=self.device)

        for layer_idx, layer in enumerate(self.layers):
            k_cache, v_cache = kv_manager.get_layer_cache(layer_idx)
            x = self.forward_layer(
                x, layer, k_cache, v_cache,
                block_table_tensor, cache_seqlens, kv_manager.block_size
            )

        x = self.norm(x)
        logits = self.lm_head(x)
        
        # 6. Greedy decode the next token for every request in the batch
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        return next_tokens
