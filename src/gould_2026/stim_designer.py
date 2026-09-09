import time
import numpy
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxopt import ScipyBoundedMinimize, LBFGS, ScipyMinimize
import itertools
import copy
import warnings
from enum import Enum

def _identity_u_to_s(u):
    return u

def _linear_u_to_s(u, A, stim_magnitude):
    return stim_magnitude * (A @ u)

def _kernel_reg_u_to_s(u, stim_magnitude, f, pred, current_t):
    return stim_magnitude * f([pred, u, current_t])

class OptimizationMethod(str, Enum):
    JAXOPT = 'jaxopt'
    JAXOPT_UNCONSTRAINED = 'jaxopt_unconstrained'
    JAXOPT_POSITIVE_CONSTRAINED = 'jaxopt_positive_constrained'
    JAXOPT_SPARSE_CONSTRAINED = 'jaxopt_sparse_constrained'
    PREV_SEEN = 'prev_seen'
    CHEAT_LOWD_VEC = 'cheat_lowd_vec'
    CHEAT_HIGHD_VEC_SINGLE_NEURONS = 'cheat_highd_vec_single_neurons'
    CHEAT_HIGHD_VEC_MANY_NEURONS = 'cheat_highd_vec_many_neurons'  # TODO: this isn't really cheating, change the name?


class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=0,  # TODO: make this an rng
            should_log=False,
            lam_1=0.001,
            optimization_method=OptimizationMethod.JAXOPT,
            u_to_s_model_type='identity', # TODO: remove? it's used in sim_stim_design_stim
            n_random_initialization=1,
    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        assert max_l0_norm > 0
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1
        self.u_to_s_model_type = u_to_s_model_type

        self.optimization_method: OptimizationMethod = optimization_method
        self.n_random_initialization = n_random_initialization

        self.log = []



    def design_stim_prev_seen(self, v, previous_us, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = _identity_u_to_s

        # TODO: keep this consistent with jaxopt version
        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = 0
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        best_u = None
        best_loss = float('inf')
        # TODO: parallellize this
        for u in previous_us:
            loss = objective(u)
            if loss < best_loss:
                best_loss = loss
                best_u = u

        best_u = best_u / best_u.max()
        return best_u, {'s': u_to_s_function(u)}

    def design_stim_jaxopt(self, v, u_dimension, rng, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = _identity_u_to_s

        u = rng.uniform(size=(u_dimension,)) * .1

        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        lb = jnp.zeros_like(u)
        ub = jnp.ones_like(u)

        bounds = (lb, ub)
        intermediate_xs = []
        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
        result = runner.run(u, bounds=bounds)
        u = numpy.array(result.params)

        if u.max() > 0:
            u = numpy.array(u / u.max())


        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0

        return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}

    def design_stim_jaxopt_generalized(self, v, u_dimension, rng, u_to_s_function=None, sparse_constrained=True, positive_constrained=True):
        if u_to_s_function is None:
            u_to_s_function = _identity_u_to_s

        if positive_constrained:
            old_rng = copy.deepcopy(rng)
            u = rng.uniform(size=(u_dimension,)) * .1 # to replicate later
        else:
            u = rng.normal(size=(u_dimension,)) * 1 / (10 * numpy.sqrt(12))


        if sparse_constrained:
            def objective(u):
                s = u_to_s_function(u)
                s_norm = jnp.linalg.norm(s)
                loss = 0
                loss += self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
                loss += jnp.dot(s, v) / (s_norm + 1e-10)
                return -loss.reshape()
        else:
            def objective(u):
                s = u_to_s_function(u)
                s_norm = jnp.linalg.norm(s)
                loss = 0
                loss += jnp.dot(s, v) / (s_norm + 1e-10)
                return -loss.reshape()

        intermediate_xs = []

        if positive_constrained:
            lb = jnp.zeros_like(u)
            ub = jnp.ones_like(u)
            bounds = (lb, ub)
            runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
            result = runner.run(u, bounds=bounds)
        else:
            runner = ScipyMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
            # runner = LBFGS(fun=objective)
            result = runner.run(u)

        u = numpy.array(result.params)

        if (m := numpy.abs(u).max()) > 0:
            u = numpy.array(u / m)

        if sparse_constrained:
            idx = numpy.argsort(u)
            u[idx[:-self.max_l0_norm]] = 0

        if sparse_constrained and positive_constrained:
            u_2,l = self.design_stim_jaxopt(v, u_dimension, old_rng, u_to_s_function=u_to_s_function)
            assert numpy.allclose(u, u_2)

        return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}


    # def design_stim_jaxopt_unconstrained(self, v, u_dimension, u_to_s_function=None):
    #     if u_to_s_function is None:
    #         u_to_s_function = _identity_u_to_s
    #
    #     u = self.rng.normal(size=(u_dimension,)) * 1/numpy.sqrt(12)
    #
    #     def objective(u):
    #         s = u_to_s_function(u)
    #         s_norm = jnp.linalg.norm(s)
    #         loss = 0
    #         # loss += self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
    #         loss += jnp.dot(s, v) / (s_norm + 1e-10)
    #         return -loss.reshape()
    #
    #     # lb = jnp.zeros_like(u)
    #     # ub = jnp.ones_like(u)
    #     #
    #     # bounds = (lb, ub)
    #     intermediate_xs = []
    #     # runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
    #     # result = runner.run(u, bounds=bounds)
    #
    #     runner = LBFGS(fun=objective)
    #     result = runner.run(u)
    #     u = numpy.array(result.params)
    #
    #     if numpy.abs(u).max() > 0:
    #         u = numpy.array(u / numpy.abs(u).max())
    #
    #
    #     # idx = numpy.argsort(u)
    #     # u[idx[:-self.max_l0_norm]] = 0
    #
    #     return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}

    # def design_stim_jaxopt_positive_constrained(self, v, u_dimension, u_to_s_function=None):
    #     if u_to_s_function is None:
    #         u_to_s_function = _identity_u_to_s
    #
    #     u = self.rng.uniform(size=(u_dimension,)) * .1
    #
    #     def objective(u):
    #         s = u_to_s_function(u)
    #         s_norm = jnp.linalg.norm(s)
    #         loss = 0
    #         loss += jnp.dot(s, v) / (s_norm + 1e-10)
    #         return -loss.reshape()
    #
    #     lb = jnp.zeros_like(u)
    #     ub = jnp.ones_like(u)
    #
    #     bounds = (lb, ub)
    #     intermediate_xs = []
    #     runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
    #     result = runner.run(u, bounds=bounds)
    #
    #     u = numpy.array(result.params)
    #
    #     if numpy.abs(u).max() > 0:
    #         u = numpy.array(u / numpy.abs(u).max())
    #
    #     return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}
    #
    # def design_stim_jaxopt_sparse_constrained(self, v, u_dimension, u_to_s_function=None):
    #     if u_to_s_function is None:
    #         u_to_s_function = _identity_u_to_s
    #
    #     u = self.rng.normal(size=(u_dimension,)) * 1/numpy.sqrt(12)
    #
    #     def objective(u):
    #         s = u_to_s_function(u)
    #         s_norm = jnp.linalg.norm(s)
    #         loss = 0
    #         loss += self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
    #         loss += jnp.dot(s, v) / (s_norm + 1e-10)
    #         return -loss.reshape()
    #
    #     intermediate_xs = []
    #     runner = LBFGS(fun=objective)
    #     result = runner.run(u)
    #     u = numpy.array(result.params)
    #
    #     if numpy.abs(u).max() > 0:
    #         u = numpy.array(u / numpy.abs(u).max())
    #
    #
    #     idx = numpy.argsort(u)
    #     u[idx[:-self.max_l0_norm]] = 0
    #
    #     return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}



    def design_stim(self, v, optimization_method=None, **kwargs):
        start_time = time.perf_counter()
        assert len(v.shape) == 2

        l = {}
        if optimization_method is None:
            optimization_method = self.optimization_method

        match optimization_method:
            case OptimizationMethod.JAXOPT:
                u, l = self.design_stim_jaxopt(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], rng=self.rng)

                # import warnings
                # warnings.warn("calling slow jaxopt_generalized")
                # u, l = self.design_stim_jaxopt_generalized(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], rng=self.rng, sparse_constrained=True, positive_constrained=True)
            case OptimizationMethod.JAXOPT_UNCONSTRAINED:
                u, l = self.design_stim_jaxopt_generalized(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], rng=self.rng, sparse_constrained=False, positive_constrained=False)
            case OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED:
                u, l = self.design_stim_jaxopt_generalized(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], rng=self.rng, sparse_constrained=False, positive_constrained=True)
            case OptimizationMethod.JAXOPT_SPARSE_CONSTRAINED:
                u, l = self.design_stim_jaxopt_generalized(v, u_dimension=kwargs['u_dimension'], u_to_s_function=kwargs['u_to_s_function'], rng=self.rng, sparse_constrained=True, positive_constrained=False)
            case OptimizationMethod.PREV_SEEN:
                u, l = self.design_stim_prev_seen(v, kwargs['previous_us'], kwargs['u_to_s_function'])
            case OptimizationMethod.CHEAT_LOWD_VEC:
                u = (kwargs['equivalent_projection_matrix'] @ v).flatten()
            case OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0])] = 1
            case OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0], size=self.max_l0_norm, replace=False)] = 1
            case _:
                raise ValueError()


        if self.should_log:
            self.log.append({
                'optimization_time': time.perf_counter() - start_time,
                'v':v,
                'u':u,
                's': numpy.nan * v
            } | l)

        return u


    def add_to_last_log(self, d:dict, assert_callback=lambda l: True):
        if not self.should_log:
            return
        assert assert_callback(self.log[-1])
        self.log[-1].update(d)