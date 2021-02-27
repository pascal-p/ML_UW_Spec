## utils for this C03 notebooks
import functools
import time
import json


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


def get_numpy_data(data_sf, features, label):
    data_sf['intercept'] = 1.
    features = ['intercept', *features]
    features_sf = data_sf[features]
    feature_matrix = features_sf.to_numpy()
    label_sarray = data_sf[label]
    label_array = label_sarray.to_numpy()
    return (feature_matrix, label_array)


def load_important_words(json_file='../data/important_words.json'):
    with open('../data/important_words.json', 'r') as f: # Reads the list of most frequent words
        important_words = json.load(f)
    return important_words

