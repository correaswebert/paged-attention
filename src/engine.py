from processor import Processor
from scheduler import Scheduler


def main():
    scheduler = Scheduler()
    processor = Processor(scheduler)

    try:
        prompt = input(">>> ")
        processor.process(prompt)
        
        # Start execution loop for submitted requests
        while scheduler.waitq or scheduler.runq:
            scheduler.schedule()
            scheduler.step()
        

    except KeyboardInterrupt:
        pass
