import asyncio
import queue
import threading

import torch
import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cache_manager import KVCacheManager
from model import PagedAttentionModel
from processor import Processor
from request import InferenceRequest
from scheduler import Scheduler

# Check CUDA availability early
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for flash_attn_with_kvcache.")

MODEL_ID = "Qwen/Qwen1.5-0.5B"
TOKENS_PER_BLOCK = 16
NUMBER_OF_BLOCKS = 128

processor = Processor(MODEL_ID)
model = PagedAttentionModel(MODEL_ID)
kv_manager = KVCacheManager(
    num_layers=len(model.layers),
    num_blocks=NUMBER_OF_BLOCKS,
    block_size=TOKENS_PER_BLOCK,
    num_heads=model.num_kv_heads,
    head_dim=model.head_dim,
)

scheduler = Scheduler(model, kv_manager, processor)

# Run the continuous processing loop in the background
threading.Thread(target=scheduler.process_loop, daemon=True).start()

app = FastAPI()


@app.post("/chat")
async def chat_endpoint(prompt: str):
    tokens = processor.encode(prompt)
    out_queue = queue.Queue()

    request = InferenceRequest(tokens, out_queue)
    scheduler.add_request(request)

    async def stream_generator():
        while True:
            # Non-blocking wait for the background thread to produce a token
            while out_queue.empty():
                await asyncio.sleep(0.01)

            chunk = out_queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")


def main(port: int = 8000):
    print("Starting Paged Attention Engine...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
