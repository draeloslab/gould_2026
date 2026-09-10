import time
import numpy
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxopt import ScipyBoundedMinimize, LBFGS, ScipyMinimize
import copy
from enum import Enum

def _objective(u, v, u_to_s_function, lam_1, max_l0_norm, eps1=0., eps2=1e-10):
    s = u_to_s_function(u)
    s_norm = jnp.sqrt(jnp.sum(jnp.square(s)) + eps1)
    loss = 0
    loss += jnp.where(lam_1 == 0.0, 0.0, lam_1 * (max_l0_norm - jnp.sum(jnp.abs(u))))
    loss += jnp.dot(s, v) / (s_norm + eps2)
    return -loss.reshape()


def _identity_u_to_s(u):
    return u

def _linear_u_to_s(u, A, stim_magnitude):
    return stim_magnitude * (A @ u)

def _kernel_reg_u_to_s(u, stim_magnitude, f, pred, current_t):
    return stim_magnitude * f([pred, u, current_t])

class OptimizationMethod(str, Enum):
    LBFGS = 'lbfgs'
    LBFGS_UNCONSTRAINED = 'lbfgs_unconstrained'
    LBFGS_POSITIVE_CONSTRAINED = 'lbfgs_positive_constrained'
    LBFGS_SPARSE_CONSTRAINED = 'lbfgs_sparse_constrained'
    # PREV_SEEN = 'prev_seen'
    CHEAT_LOWD_VEC = 'cheat_lowd_vec'
    RANDOM_SINGLE_NEURONS = 'cheat_highd_vec_single_neurons'
    RANDOM_MANY_NEURONS = 'cheat_highd_vec_many_neurons'


class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=0,  # TODO: make this an rng
            should_log=False,
            lam_1=0.001,
            optimization_method=OptimizationMethod.LBFGS,
            min_radius=0.,
            eps1=10 ** -4.5,
            eps2=1e-5,
            n_random=1,
            n_previous=50,

    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        assert max_l0_norm > 0
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1
        self.min_radius = min_radius
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_random = n_random
        self.n_previous = n_previous

        self.optimization_method: OptimizationMethod = optimization_method

        self._box_constrained_optimizer = ScipyBoundedMinimize(fun=_objective, method='l-bfgs-b')
        self._box_unconstrained_optimizer = ScipyMinimize(fun=_objective, method='l-bfgs-b')


        self.log = []

    def design_stim_lbfgs(self, v, u_dimension, rng, u_to_s_function=None, previous_us=None, sparse_constrained=True, positive_constrained=True):
        if u_to_s_function is None:
            u_to_s_function = _identity_u_to_s

        if sparse_constrained:
            lam_1 = self.lam_1
        else:
            lam_1 = 0

        us_to_try = []
        for _ in range(self.n_random):
            if positive_constrained:
                u = rng.uniform(size=(u_dimension,)) * .1
            else:
                u = rng.normal(size=(u_dimension,)) * 1 / (10 * numpy.sqrt(12))
            us_to_try.append(u)

        previous_performances = []
        if self.n_previous > 0 and previous_us is not None:
            for u in previous_us:
                previous_performances.append(_objective(u,v,u_to_s_function,lam_1, self.max_l0_norm, eps1=self.eps1, eps2=self.eps2))
            for i in numpy.argsort(previous_performances)[:self.n_previous]:
                us_to_try.append(previous_us[i])


        results = []
        previous_performances = []
        for u in us_to_try:
            previous_performances.append(_objective(u, v, u_to_s_function, lam_1, self.max_l0_norm, eps1=self.eps1, eps2=self.eps2))

            ub = jnp.ones_like(u)
            lb = jnp.zeros_like(u) if positive_constrained else -jnp.ones_like(u)
            bounds = (lb, ub)
            result = self._box_constrained_optimizer.run(u, bounds=bounds, v=v, u_to_s_function=u_to_s_function, lam_1=lam_1, max_l0_norm=self.max_l0_norm, eps1=self.eps1, eps2=self.eps2)

            u = numpy.array(result.params)

            if sparse_constrained:
                idx = numpy.argsort(numpy.abs(u))
                u[idx[:-self.max_l0_norm]] = 0

            if (m := numpy.abs(u).max()) > 0:
                u = numpy.array(u / m)

            results.append(u)

        s_s = [u_to_s_function(u) for u in results]
        losses = [_objective(u, v, u_to_s_function, lam_1, self.max_l0_norm, eps1=self.eps1, eps2=self.eps2) for u in results]
        radii = [numpy.linalg.norm(s) for s in s_s]

        best_idx = 0
        best_loss = losses[0]
        for i, (loss, radius) in enumerate(zip(losses, radii)):
            if loss < best_loss and radius > self.min_radius:
                best_idx = i
                best_loss = loss
        u = results[best_idx]

        return u, {'s': u_to_s_function(u), 'v': v, 'sparse_constrained': sparse_constrained, 'positive_constrained': positive_constrained, 'previous_performances': previous_performances, 'radii':radii, 'best_idx':best_idx, 'losses': losses}



    def design_stim(self, v, optimization_method=None, **kwargs):
        start_time = time.perf_counter()
        assert len(v.shape) == 2

        l = {}
        if optimization_method is None:
            optimization_method = self.optimization_method

        match optimization_method:
            case OptimizationMethod.LBFGS:
                u, l = self.design_stim_lbfgs(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], previous_us=kwargs['previous_us'], rng=self.rng, sparse_constrained=True, positive_constrained=True)
            case OptimizationMethod.LBFGS_UNCONSTRAINED:
                u, l = self.design_stim_lbfgs(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], previous_us=kwargs['previous_us'], rng=self.rng, sparse_constrained=False, positive_constrained=False)
            case OptimizationMethod.LBFGS_POSITIVE_CONSTRAINED:
                u, l = self.design_stim_lbfgs(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], previous_us=kwargs['previous_us'], rng=self.rng, sparse_constrained=False, positive_constrained=True)
            case OptimizationMethod.LBFGS_SPARSE_CONSTRAINED:
                u, l = self.design_stim_lbfgs(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], previous_us=kwargs['previous_us'], rng=self.rng, sparse_constrained=True, positive_constrained=False)
            case OptimizationMethod.CHEAT_LOWD_VEC:
                u = (kwargs['equivalent_projection_matrix'] @ v).flatten()
            case OptimizationMethod.RANDOM_SINGLE_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0])] = 1
            case OptimizationMethod.RANDOM_MANY_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0], size=self.max_l0_norm, replace=False)] = 1
            case _:
                raise ValueError()


        if self.should_log:
            self.log.append({
                'optimization_time': time.perf_counter() - start_time,
                'v':v,
                'u':u,
                's': numpy.nan * v,
                'optimization_method': optimization_method,
            } | l)

        return u


    def add_to_last_log(self, d:dict, assert_callback=lambda l: True):
        if not self.should_log:
            return
        assert assert_callback(self.log[-1])
        self.log[-1].update(d)