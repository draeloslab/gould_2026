import jax
import numpy
import jax.numpy as jnp
import warnings

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


class MultiKernelRegressor:
    def __init__(self, length_scales=(1e-1,1e-1,1e-9), maxlen=100, input_names=('stim_location', 'stim_vector', 'stim_time'), reweight_every=1, rng=None):
        self.maxlen = maxlen
        self.input_histories = None
        self.output_history = None
        self.n_observed = 0
        self.input_names = input_names
        self.reweight_every = reweight_every
        if rng is None:
            rng = numpy.random.default_rng(0)
        self.rng = rng
        self.log = {'length_scales': [], 'preq_errors':[]}

        self.length_scales = numpy.array(length_scales)

    def observe(self, x, y):
        if any([numpy.any(~numpy.isfinite(sub_x)) for sub_x in x]) or numpy.any(~numpy.isfinite(y)):
            warnings.warn("ignoring non-finite input")
            return

        self.log['preq_errors'].append(y - self.predict(x))

        if self.input_histories is None:
            self.input_histories = [numpy.zeros(shape=(self.maxlen, sub_x.size)) * numpy.nan for sub_x in x]
            self.output_history = numpy.zeros(shape=(self.maxlen, y.size))

        if self.n_observed == self.maxlen:
            warnings.warn("history is full, overwriting old observations")
        index = self.n_observed % self.maxlen

        for history, sub_x in zip(self.input_histories, x):
            history[index, :] = sub_x
        self.output_history[index, :] = y
        self.n_observed += 1

        if self.n_observed % self.reweight_every == 0:
            self.reweight()

    def reweight(self):
        sample_size = min(self.n_observed, 15)
        sample = self.rng.permutation(min(self.n_observed, self.maxlen))[:sample_size]
        log_external_weight_vec = numpy.zeros(self.maxlen)
        log_external_weight_vec[sample] = -numpy.inf
        f = self.make_jax_pred_f()
        def evaluate(length_scales):
            if numpy.any(length_scales <= 1e-10) or numpy.any(length_scales > 1e6):
                return numpy.inf

            errors = numpy.zeros(sample_size)
            for i, idx in enumerate(sample):
                try:
                    errors[i] = numpy.linalg.norm(f([h[idx] for h in self.input_histories], length_scales, log_external_weight_vec=log_external_weight_vec) - self.output_history[idx])**2
                except (OverflowError, ZeroDivisionError):
                    errors[i] = numpy.inf
            return numpy.mean(errors)

        current = evaluate(self.length_scales)
        new_length_scales = numpy.array(self.length_scales)

        coefs = numpy.logspace(-1, 1, 5)
        for i in range(len(self.length_scales)):
            errors = numpy.zeros(5)
            for j, coef in enumerate(coefs):
                if coef == 1:
                    errors[j] = current
                    continue
                test_length_scales = numpy.array(self.length_scales)
                test_length_scales[i] *= coef
                errors[j] = evaluate(test_length_scales)
            new_length_scales[i] *= coefs[numpy.argmin(errors)]

        self.log['length_scales'].append(numpy.array(self.length_scales))
        lr = 0.05
        self.length_scales = numpy.exp(numpy.log(self.length_scales) * lr + numpy.log(new_length_scales) * (1-lr))

    def plot_length_scales(self, ax):
        for series, label in zip(numpy.array(self.log['length_scales']).T, self.input_names):
            ax.plot(series, label=label + ' curvy')
        ax.semilogy()


    def make_jax_pred_f(self):
        # TODO: precompute
        if self.input_histories is None:
            def f(x):
                return numpy.array([[numpy.nan]])
        else:
            input_histories = [jnp.array(h) for h in self.input_histories]
            output_history = jnp.array(self.output_history)
            zeros = jnp.zeros(len(self.output_history))
            def f(x, length_scales=jnp.array(self.length_scales), log_external_weight_vec=zeros):
                # log_external_weight_vec is for cross-validation
                distances = [-length_scale * jnp.linalg.norm(history - jnp.squeeze(sub_x), axis=1) ** 2 for
                             (sub_x, history, length_scale) in zip(x, input_histories, length_scales)]
                log_weights = jnp.array(distances).sum(axis=0)
                log_weights = jnp.nan_to_num(log_weights, nan=-numpy.inf)
                log_weights = log_weights + log_external_weight_vec
                log_sum = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_sum

                return jnp.exp(log_weights) @ output_history
        return f

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
