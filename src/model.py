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
        self.lm_head = hf_model.lm_head

        self.num_heads = hf_model.config.num_attention_heads
        self.num_kv_heads = hf_model.config.num_key_value_heads
        self.head_dim = hf_model.config.hidden_size // self.num_heads

    @torch.no_grad()
    def forward_batch(self, batch_tokens: list[int], batch_positions: list[int], batch_block_tables: list[list[int]], kv_manager) -> list[int]:
        batch_size = len(batch_tokens)
        
        # 1. Embed the whole batch at once: [batch_size, 1, head_dim]
        x = self.embed(torch.tensor(batch_tokens, device=self.device).unsqueeze(1))
        
        # 2. Extract Q, K, V
        q_raw = self.layer.self_attn.q_proj(x)
        # q = q_raw.view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.layer.self_attn.k_proj(x).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v = self.layer.self_attn.v_proj(x).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        # 3. Pad block tables so they can be converted to a square tensor
        max_blocks = max(len(t) for t in batch_block_tables)
        padded_tables = [t + [0]*(max_blocks - len(t)) for t in batch_block_tables]

        block_table_tensor = torch.tensor(padded_tables, dtype=torch.int32, device=self.device)
        cache_seqlens = torch.tensor(batch_positions, dtype=torch.int32, device=self.device)

        # 4. Custom CUDA Execute Paged Attention
        paged_attn.update(
            kv_manager.k_cache,
            kv_manager.v_cache,
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            block_table_tensor,
            kv_manager.block_size,
            cache_seqlens
        )

        # print("q", q.contiguous().shape)
        print("k", kv_manager.k_cache.shape)
        print("v", kv_manager.v_cache.shape)
        print("block_table", block_table_tensor.shape)
        print(kv_manager.block_size)
        print(cache_seqlens + 1)
        print(self.num_heads)

        attn_out = paged_attn.decode(
            q_raw.contiguous(),
            kv_manager.k_cache,
            kv_manager.v_cache,
            block_table_tensor,
            kv_manager.block_size,
            cache_seqlens + 1,
            self.num_heads,
            True
        )

        # 5. Output Projection
        attn_out = attn_out.view(batch_size, 1, -1)
        out = self.layer.self_attn.o_proj(attn_out)
        
        logits = self.lm_head(x + out)
        
        # 6. Greedy decode the next token for every request in the batch
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        return next_tokens
