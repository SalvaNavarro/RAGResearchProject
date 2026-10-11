from contextlib import contextmanager
import time

@contextmanager
def profiler(id):
    "A helper method to profile task execution time."
    begin = time.perf_counter()
    yield
    delta = time.perf_counter() - begin
    print(f"Task: {id} completed in {delta:.2f}s")