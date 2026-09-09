import jax
import numpy
import jax.numpy as jnp
import warnings
from jax.tree_util import Partial
import optax
import time


@jax.jit
def rank_one_update_formula1(D, x1, x2):
    # TODO: maybe this is only faster if we put it on the GPU? maybe move the data?
    # TODO: if this is called multiple times, maybe define this in-function so there isn't a lookup process?
    return D - (D @ x1 @ x2.T @ D) / (1 + x2.T @ D @ x1)



class VanillaOnlineRegressor:
    def __init__(self, init_min_ratio=1.1, add_intercept=True, regularization_factor=0.01):
        self.add_intercept = add_intercept
        self.init_min_ratio = init_min_ratio
        self.regularization_factor = regularization_factor

        # core stuff
        self.input_d = None
        self.output_d = None
        self.D = None  # this should be None for a while
        self.F = None
        self.c = None

        # initializations
        self.n_observed = 0

    def format_x(self, x):
        x = x.reshape([-1, 1])
        if self.add_intercept:
            x = numpy.vstack([x, [1]])
        return x

    def _observe(self, x, y, update_D=False):
        x = self.format_x(x)
        y = numpy.squeeze(y)

        if update_D:
            self.D = rank_one_update_formula1(self.D, x, x)
        else:
            self.F = self.F + x @ x.T
        self.c = self.c + x*y

        self.n_observed += 1

    def observe(self, x, y):
        if numpy.any(~numpy.isfinite(x)) or numpy.any(~numpy.isfinite(y)):
            return

        # x and y should be vectors
        if self.F is None and self.c is None:  # this is the first observation
            self.input_d = x.size + self.add_intercept
            self.output_d = y.size
            if self.regularization_factor == 0:
                self.F = numpy.zeros([self.input_d, self.input_d])
                self.c = numpy.zeros([self.input_d, self.output_d])
            else:
                self.D = numpy.eye(self.input_d) / self.regularization_factor
                self.c = numpy.zeros([self.input_d, self.output_d])

        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self._observe(x, y, update_D=True)
        else:
            self._observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                # initialize
                self.D = numpy.linalg.pinv(self.F)

    def get_beta(self):
        if self.c is None:
            return numpy.nan

        if self.D is None:
            return numpy.zeros((self.input_d, self.output_d)) * numpy.nan
        return self.D @ self.c

    def predict(self, x):
        if self.c is None:
            return numpy.array(numpy.nan)

        x = self.format_x(x)
        beta = self.get_beta()

        return (x.T @ beta).flatten()


@jax.jit
def _predict(x, input_histories, output_history, valid, log_length_scales):
    log_weights = 0.0
    for (sub_x, history, length_scale) in zip(x, input_histories, jnp.exp(log_length_scales)):
        diffs = history - jnp.squeeze(sub_x)
        sq_distances = jnp.sum(jnp.square(diffs), axis=1)
        log_weights = log_weights - length_scale * sq_distances

    log_weights = jnp.where(valid, log_weights, -jnp.inf)
    log_sum = jax.scipy.special.logsumexp(log_weights)
    log_weights = log_weights - log_sum

    return jnp.exp(jnp.clip(log_weights, max=0, min=-30)) @ output_history

def _loo_squared_error(idx, log_length_scales, input_histories, output_history, valid):
    x_i = [h[idx] for h in input_histories]
    y_i = output_history[idx]
    v = valid.at[idx].set(False)
    y_hat = _predict(x=x_i, input_histories=input_histories, output_history=output_history, valid=v, log_length_scales=log_length_scales)
    return jnp.sum(jnp.square(y_hat - y_i))

_batched_loo_squared_error = jax.vmap(_loo_squared_error, in_axes=(0, None, None, None, None))
def _loo_mse(sample, log_length_scales, input_histories, output_history, valid):
    return jnp.mean(_batched_loo_squared_error(sample, log_length_scales, input_histories, output_history, valid))

_loo_mse_and_grad = jax.jit(jax.value_and_grad(_loo_mse, argnums=1))


class KernelRegressor:
    def __init__(self, length_scales=(1e-1,1e-1,1e-9), maxlen=100, input_names=('stim_location', 'stim_vector', 'stim_time'), reweight_every=1, rng=None, adam_lr=0.01, log_level=0):
        self.maxlen = maxlen
        self.input_histories = None
        self.output_history = None
        self.n_observed = 0
        self.input_names = input_names
        self.reweight_every = reweight_every
        if rng is None:
            rng = numpy.random.default_rng(0)
        self.rng = rng
        self.log_level = log_level
        self.log = {'log_length_scales': [], 'preq_errors':[], 'reweight_times': []}

        self.log_length_scales = jnp.array(numpy.log(length_scales), dtype=jnp.float32)

        self.loo_max_sample_size = 30
        self._opt = optax.adam(learning_rate=adam_lr)
        self._opt_state = self._opt.init(self.log_length_scales)

    def observe(self, x, y):
        if any([numpy.any(~numpy.isfinite(sub_x)) for sub_x in x]) or numpy.any(~numpy.isfinite(y)):
            return

        if self.log_level >= 2:
            self.log['preq_errors'].append(y - self.predict(x))

        if self.input_histories is None:
            self.input_histories = [numpy.zeros(shape=(self.maxlen, sub_x.size)) * numpy.nan for sub_x in x]
            self.output_history = numpy.zeros(shape=(self.maxlen, y.size))

        index = self.n_observed % self.maxlen

        for history, sub_x in zip(self.input_histories, x):
            history[index, :] = sub_x
        self.output_history[index, :] = y
        self.n_observed += 1

        if self.n_observed % self.reweight_every == 0:
            self.reweight()

    def reweight(self):
        start_time = time.monotonic_ns()

        if self.n_observed < 5:
            return
        sample_size = min(self.n_observed, self.loo_max_sample_size)
        sample = self.rng.permutation(min(self.n_observed, self.maxlen))[:sample_size]
        _, batched_loo_mse_and_grad = self._bind_eval_fns()

        loss, grad = batched_loo_mse_and_grad(sample, self.log_length_scales)
        updates, self._opt_state = self._opt.update(grad, self._opt_state, self.log_length_scales)
        self.log_length_scales = optax.apply_updates(self.log_length_scales, updates)
        self.log['log_length_scales'].append(numpy.array(self.log_length_scales))

        duration = time.monotonic_ns() - start_time
        self.log['reweight_times'].append(duration / 1e9)  # convert to seconds



    def plot_length_scales(self, ax):
        for series, label in zip(numpy.array(self.log['log_length_scales']).T, self.input_names):
            ax.plot(series)


    def _bind_eval_fns(self):
        if self.input_histories is None:
            def h(x):
                return numpy.array([[numpy.nan]])
            f = h
            loo_mse_and_grad = lambda sample, log_length_scales: (numpy.nan, numpy.zeros_like(log_length_scales))

        else:
            valid = numpy.arange(self.maxlen)
            valid = valid < self.n_observed

            input_histories = [jnp.nan_to_num(h, nan=0.0) for h in self.input_histories]
            output_history = jnp.array(self.output_history)

            f = Partial(_predict, input_histories=input_histories, output_history=output_history, valid=valid, log_length_scales=self.log_length_scales)
            loo_mse_and_grad = lambda sample, log_length_scales=self.log_length_scales: _loo_mse_and_grad(sample, log_length_scales, input_histories, output_history, valid)

        return f, loo_mse_and_grad

    def make_jax_pred_f(self):
        return self._bind_eval_fns()[0]

    def predict(self, x):
        return numpy.array(self.make_jax_pred_f()(x))

    def get_obs(self, i=None, t=None):
        """gets last by default"""
        if t is not None: # use time
            assert i is None
            candidates = numpy.nonzero(numpy.abs(t - self.input_histories[self.input_names.index('stim_time')].flatten()) < 1e-12)
            assert len(candidates) == 1
            assert len(candidates[0]) == 1
            i = candidates[0][0]
        else: # use i
            if i is None: # get last obs
                i = (self.n_observed - 1) % self.maxlen
        return {k:v[i] for k, v in zip(self.input_names, self.input_histories)} | {'output': self.output_history[i]}
