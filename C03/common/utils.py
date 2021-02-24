## utils for this C03 notebooks
import functools
import time

## decorator
def time_it(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        s_t = time.time()
        res = fn(*args, **kwargs)
        e_t = time.time()
        print(f"took: {(e_t - s_t) * 1000:3.3f}ms")
        return res

    return wrapped_fn


def assert_all(fn, coll, msg=""):
    if len(msg)> 0:
        assert all(map(fn, coll)), msg
    else:
        assert all(map(fn, coll))
    return
