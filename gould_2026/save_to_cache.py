import functools
import hashlib
import inspect
import json
import dill as pickle
import humanize
import filelock
import time
import pathlib
from abc import ABCMeta
import xxhash
from dataclasses import is_dataclass, asdict

import numpy as np

from gould_2026.estimator import ArrayWithTime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Store a compact fingerprint instead of the full array contents
            return {
                '__ndarray__': True,
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'hash': hashlib.sha1(obj.tobytes()).hexdigest(),
            }
        elif isinstance(obj, ArrayWithTime):
            return (self.default(obj.as_array()), self.default(obj.t))
        elif isinstance(obj, np.random.Generator):
            return (obj.bit_generator.__class__, obj.bit_generator.state)
        elif is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, ABCMeta):
            return inspect.getsource(obj)
        return json.JSONEncoder.default(self, obj)


def _hash_value(x):
    """Recursively build a hashlib.sha1 digest that handles large numpy arrays efficiently."""
    h = xxhash.xxh64()
    if isinstance(x, np.ndarray):
        h.update(str(x.shape).encode())
        h.update(str(x.dtype).encode())
        # Hash raw bytes directly — no copy, no list conversion
        h.update(np.ascontiguousarray(x).data)
        if isinstance(x, ArrayWithTime):
            h.update(np.ascontiguousarray(x.t).data)
    elif isinstance(x, dict):
        for k in sorted(x.keys(), key=str):
            h.update(str(k).encode())
            h.update(_hash_value(x[k]).digest())
    elif isinstance(x, (list, tuple)):
        for item in x:
            h.update(_hash_value(item).digest())
    elif is_dataclass(x):
        h.update(_hash_value(asdict(x)).digest())
    elif isinstance(x, np.random.Generator):
        h.update(json.dumps((str(x.bit_generator.__class__), x.bit_generator.state), sort_keys=True).encode())
    else:
        h.update(json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode())
    return h


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
            all_args_as_key = _hash_value(all_args).hexdigest()

            if _recalculate_cache_value or all_args_as_key not in cache_index or not (location/ cache_index[all_args_as_key]['cache_file']).exists():
                start = time.perf_counter()
                result = original_function(**all_args)
                execute_time = time.perf_counter() - start

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

            to_load_from = cache_index[all_args_as_key]['cache_file']
            with open(to_load_from, 'rb') as fhan:
                print(f"retreiving cache from: {to_load_from}")
                return pickle.load(fhan)

        return new_function

    return decorator
