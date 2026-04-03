import functools
import hashlib
import inspect
import json
import dill as pickle
import humanize
import filelock
import time
import pathlib

import numpy as np

from gould_2026.estimator import ArrayWithTime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ArrayWithTime):
            return [obj.tolist(), obj.t.tolist()]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.random.Generator):
            return (obj.bit_generator.__class__, obj.bit_generator.state)
        return json.JSONEncoder.default(self, obj)


def make_hashable(x):
    return json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode()


def make_hashable_and_hash(x):
    return int(hashlib.sha1(make_hashable(x)).hexdigest(), 16)


def save_to_cache(file, location):
    location = pathlib.Path(location)

    def decorator(original_function):
        @functools.wraps(original_function)
        def new_function(*args, _recalculate_cache_value=False, **kwargs):
            cache_index_file = (location / f"{file}_index.json").resolve()
            try:
                with open(cache_index_file, 'r') as fhan:
                    cache_index = json.load(fhan)
            except FileNotFoundError:
                cache_index = {}

            bound_args = inspect.signature(original_function).bind(*args, **kwargs)
            bound_args.apply_defaults()

            all_args = bound_args.arguments
            all_args_as_key = str(make_hashable_and_hash(all_args))

            if _recalculate_cache_value or all_args_as_key not in cache_index or not (location/ cache_index[all_args_as_key]['cache_file']).exists():
                start = time.time()
                result = original_function(**all_args)
                execute_time = time.time() - start

                hstring = str(all_args_as_key)[-15:]
                cache_file = str((location/ f"{file}_{hstring}.pickle").resolve())
                print(f"caching value in: {cache_file}")
                with open(cache_file, "wb") as fhan:
                    pickle.dump(result, fhan)


                cache_index[all_args_as_key] = {
                    'cache_file': cache_file,
                    'execute_time': execute_time,
                    'execute_time_human_readable': humanize.precisedelta(execute_time, minimum_unit="milliseconds"),
                    'args': str(all_args),
                    'filesize_gb': pathlib.Path(cache_file).stat().st_size/1e9,
                    'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'hits': 0
                }

            with filelock.FileLock(cache_index_file.with_suffix('.lock')):
                try:
                    with open(cache_index_file, 'r') as fhan:
                        updated_cache_index = json.load(fhan)
                except FileNotFoundError:
                    updated_cache_index = {}

                updated_cache_index.update(cache_index)

                updated_cache_index[all_args_as_key]['hits'] += 1

                with open(cache_index_file, 'w') as fhan:
                    json.dump(updated_cache_index, fhan, indent=4)

                cache_index = updated_cache_index

            to_load_from = location / cache_index[all_args_as_key]['cache_file']
            with open(to_load_from, 'rb') as fhan:
                print(f"retreiving cache from: {to_load_from}")
                return pickle.load(fhan)

        return new_function

    return decorator
