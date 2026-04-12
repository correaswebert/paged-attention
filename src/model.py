import torch
from flash_attn import flash_attn_with_kvcache
from transformers import AutoModelForCausalLM

from cache_manager import KVCacheManager


class PagedAttentionModel(torch.nn.Module):
    """Wrapper around real HF weights to execute flash_attn_with_kvcache."""

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

        self.num_heads = self.layer.self_attn.num_heads
        self.num_kv_heads = self.layer.self_attn.num_key_value_heads
        self.head_dim = self.layer.self_attn.head_dim

    @torch.no_grad()
    def forward_step(
        self,
        token_id: int,
        pos: int,
        block_table: list[int],
        kv_manager: KVCacheManager,
    ) -> int:
        x = self.embed(torch.tensor([[token_id]], device=self.device))

        q = self.layer.self_attn.q_proj(x).view(1, 1, self.num_heads, self.head_dim)
        k = self.layer.self_attn.k_proj(x).view(1, 1, self.num_kv_heads, self.head_dim)
        v = self.layer.self_attn.v_proj(x).view(1, 1, self.num_kv_heads, self.head_dim)

        block_table_tensor = torch.tensor(
            [block_table], dtype=torch.int32, device=self.device
        )
        cache_seqlens = torch.tensor([pos], dtype=torch.int32, device=self.device)

        attn_out = flash_attn_with_kvcache(
            q,
            kv_manager.k_cache,
            kv_manager.v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            block_table=block_table_tensor,
            causal=True,
        )

        attn_out = attn_out.view(1, 1, -1)
        out = self.layer.self_attn.o_proj(attn_out)

        # Simplified residual/norm
        logits = self.lm_head(x + out)

        # Greedy decoding
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        return int(next_token.real)
