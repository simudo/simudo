import timeit

__all__ = ['xtimeit']

def xtimeit(thunk, timer=timeit.default_timer,
            overall_time=1.0, warmup_iterations=3):
    n = 10
    def measurement():
        start = timer()
        thunk()
        return timer() - start
    # warm-up
    for i in range(warmup_iterations):
        thunk()
    lst = []
    start = timer()
    while timer() - start <= overall_time:
        lst.append(measurement())
    return min(lst)
