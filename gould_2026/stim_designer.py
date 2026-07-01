import time
import numpy
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize, LBFGS, ScipyMinimize
import itertools
import copy
import warnings
from enum import Enum

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
            inter_stim_interval_generator=None,
            optimization_method=OptimizationMethod.JAXOPT,
            stim_timing_method='regular',
            initial_nostim_period=1,
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
        self.stim_timing_method = stim_timing_method
        self.initial_nostim_period = initial_nostim_period

        if inter_stim_interval_generator is None:
            inter_stim_interval_generator = itertools.repeat(1)
        self.inter_stim_interval_generator = inter_stim_interval_generator
        self.last_stim_time = None
        self.current_isi = None

        self.log = []

        self.objective_history = []

    def stim_when_extreme(self, current_t, objective_value):
        self.objective_history.append(objective_value)
        return current_t > 50 and objective_value == numpy.nanmin(self.objective_history)

    def decide_whether_to_stim(self, current_t, **kwargs):
        if current_t < self.initial_nostim_period:
            return False

        if self.stim_timing_method == 'isi':  # or 'regular'
            if self.last_stim_time is None:
                self.last_stim_time = self.initial_nostim_period if self.initial_nostim_period is not None else 0
                self.current_isi = next(self.inter_stim_interval_generator)
            if current_t > self.last_stim_time + self.current_isi:
                self.last_stim_time = current_t
                self.current_isi = next(self.inter_stim_interval_generator)
                return True
            return False
        elif self.stim_timing_method == 'extreme':
            return self.stim_when_extreme(current_t, **kwargs)
        elif self.stim_timing_method == 'random':
            return kwargs['stim_time_rng'].random() < 1/next(self.inter_stim_interval_generator) * kwargs['input_array_dt']
        else:
            raise ValueError()



    def register_stim(self):
        pass

    def design_stim_prev_seen(self, v, previous_us, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = lambda u: u

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
            u_to_s_function = lambda x: x

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
            u_to_s_function = lambda x: x

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
    #         u_to_s_function = lambda x: x
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
    #         u_to_s_function = lambda x: x
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
    #         u_to_s_function = lambda x: x
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

    def sim_stim_design_stim(self, sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t):
        self: StimDesigner
        optimization_method = self.optimization_method
        u_to_s_model_type = self.u_to_s_model_type
        if sr.stim_reg.n_observed <= self.n_random_initialization and (u_to_s_model_type == 'kernel_regressed' or optimization_method == 'prev_seen'):
            # u_to_s_model_type = 'identity'
            u_to_s_model_type = None
            optimization_method = 'cheat_highd_vec_many_neurons'


        if optimization_method in {OptimizationMethod.JAXOPT, OptimizationMethod.JAXOPT_UNCONSTRAINED, OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED, OptimizationMethod.JAXOPT_SPARSE_CONSTRAINED, OptimizationMethod.PREV_SEEN}:
            stim_reg = sr.stim_reg
            previous_us = stim_reg.input_histories[1][:stim_reg.n_observed] if optimization_method == 'prev_seen' else None
            if u_to_s_model_type == 'kernel_regressed':
                f = stim_reg.make_jax_pred_f()
                pred = sr.autoreg.predict(n_steps=0)
                def u_to_s_function(u):
                    return stim_magnitude * f([pred, u, current_t])
                designed_stim = self.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
            elif u_to_s_model_type == 'identity':
                def u_to_s_function(u):
                    return stim_magnitude * equivalent_projection_matrix.T @ u
                designed_stim = self.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
        elif optimization_method == OptimizationMethod.CHEAT_LOWD_VEC and u_to_s_model_type == 'identity':
            designed_stim = self.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix)
        elif optimization_method in {OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS, OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS}:
            designed_stim = self.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix, optimization_method=optimization_method)
        else:
            raise ValueError()

        self.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)
        self.log[-1]['time_of_stim'] = current_t
        self.log[-1]['equiv_proj_mat'] = equivalent_projection_matrix

        if (designed_stim == 0).all():
            designed_stim[0] = 1e-10
            warnings.warn("Stimulus was all zero!")  # TODO: handle this better

        return designed_stim
