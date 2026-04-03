import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scheduler import Scheduler
from src.processor import Processor

def test_paged_attention():
    scheduler = Scheduler()
    # Mocking block size down to 4 for testing preemptions efficiently
    scheduler.cache_manager.block_size = 4
    processor = Processor(scheduler)
    
    print("Adding tasks via Processor...")
    processor.process("Hello")
    processor.process("World")
    
    print("Starting execution loop...")
    while scheduler.waitq or scheduler.runq:
        scheduler.schedule()
        scheduler.step()
        print(f"  -> CacheManager Free List Size: {len(scheduler.cache_manager.free_list)}")

if __name__ == "__main__":
    test_paged_attention()
