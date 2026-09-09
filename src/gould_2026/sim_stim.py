from collections import deque
from enum import Enum
import functools
from types import SimpleNamespace
from typing import NamedTuple
from dataclasses import dataclass
from itertools import cycle
import warnings
import time

import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext

from .estimator import Pipeline, CenteringEstimator, KernelSmoother, ArrayWithTime
from .stim_regressor import StimRegressor
from .dimension_reduction.ica import mmICA
from .dimension_reduction.jpca import sjPCA
from .dimension_reduction.prosvd import proSVD
from .regression import KernelRegressor
from .prediction.kalman_filter import StreamingKalmanFilter
from .stim_designer import StimDesigner, OptimizationMethod, Partial, _kernel_reg_u_to_s, _linear_u_to_s
from .save_to_cache import save_to_cache


class StimResponseType(str, Enum):
    IDENTITY = 'identity'
    FLIP = 'flip'
    HIGH_D_PERMUTED = 'high_d_permuted'

class StimResponseModelType(str, Enum):
    IDENTITY = 'identity'
    KERNEL_REGRESSED = 'kernel_regressed'
    NO_MODEL = 'no_model'

class StimDirectionType(str, Enum):
    FIRST = 'first'
    FIRST2 = 'first2'
    COL = 'col'
    RANDOM = 'random'
    RANDOM_POSITIVE = 'random+'
    RANDOM_FEASIBLE = 'random_feasible'
    ONES = 'ones'
    NEG_ONES = '-ones'



class SimulatedStimResponseCalculator:
    """Calculates the ground truth response to stimulations in the simulation."""
    def __init__(self, *, rng, true_S=StimResponseType.IDENTITY):
        if not isinstance(true_S, StimResponseType):
            true_S = StimResponseType(true_S)
            warnings.warn(f"true_S should be a StimResponseType enum, not a string. Converting to StimResponseType.")
        self.true_S = true_S
        self.rng = rng
        self.high_d_permutation = None


    def true_stim_result(self, instantaneous_stim, equivalent_projection_matrix=None):
        if equivalent_projection_matrix is None:
            assert (instantaneous_stim == 0).all()
            return instantaneous_stim

        if self.true_S in {StimResponseType.IDENTITY, StimResponseType.FLIP, StimResponseType.HIGH_D_PERMUTED}:
            full_d_transformation_matrix = self.response_alteration_matrix(equivalent_projection_matrix)
            transformed_instantaneous_stim = full_d_transformation_matrix @ instantaneous_stim
        else:
            raise ValueError(self.true_S)

        return transformed_instantaneous_stim


    def response_alteration_matrix(self, equivalent_projection_matrix):
        if equivalent_projection_matrix is None:
            return None
        if self.high_d_permutation is None:
            self.high_d_permutation = self.rng.permutation(equivalent_projection_matrix.shape[0])

        if self.true_S == StimResponseType.IDENTITY:
            transform_matrix = np.eye(equivalent_projection_matrix.shape[0])
        elif self.true_S == StimResponseType.FLIP:
            # TODO: it feels weird to have the ground truth response depend on the learned Q
            k = equivalent_projection_matrix.shape[1]
            assert k % 2 == 0
            J2 = np.array([[0, -1], [1, 0]])
            flip_matrix = np.kron(np.eye(k//2), J2)
            transform_matrix = equivalent_projection_matrix @ flip_matrix @ equivalent_projection_matrix.T + np.eye(equivalent_projection_matrix.shape[0]) - equivalent_projection_matrix @ equivalent_projection_matrix.T
        elif self.true_S == StimResponseType.HIGH_D_PERMUTED:
            transform_matrix = np.eye(equivalent_projection_matrix.shape[0])
            transform_matrix = transform_matrix[:, self.high_d_permutation]
        else:
            raise ValueError(self.true_S)

        return transform_matrix

    def true_stim_result_in_latent_space(self, instantaneous_stim, equivalent_projection_matrix):
        return equivalent_projection_matrix.T @ self.true_stim_result(instantaneous_stim, equivalent_projection_matrix)


class SimulatedStimAdder:
    def __init__(self, *, decay=.8, stim_time_delay=0):
        self.alpha = decay

        self.to_add = 0

        self.stim_time_delay = stim_time_delay
        self.stim_delay_queue = deque([0] * stim_time_delay)

    def register_stim(self, true_stim_result):
        self.stim_delay_queue.appendleft(true_stim_result)

    def run_for_X(self, data):
        self.to_add += self.stim_delay_queue.pop()
        data = data + self.to_add
        self.to_add = self.to_add * self.alpha
        return data


def calculate_equivalent_projection_matrix(pro, last_dim_red_object):
    equivalent_projection_matrix = pro.Q
    if equivalent_projection_matrix is not None:
        if last_dim_red_object is None:
            pass
        elif isinstance(last_dim_red_object, sjPCA):
            try:
                U = last_dim_red_object.get_U()
            except AttributeError: # TODO make this more elegant
                U = None
            if U is not None:
                equivalent_projection_matrix = equivalent_projection_matrix @ U
        elif isinstance(last_dim_red_object, mmICA):
            W = last_dim_red_object.W
            if W is not None:
                equivalent_projection_matrix = equivalent_projection_matrix @ W.T
        else:
            raise ValueError()
    return equivalent_projection_matrix


def desired_stim_direction(latent_d, full_d, u_to_latent_s, stim_direction_type, rng, max_l0_norm):
    numpy = np
    if not isinstance(stim_direction_type, StimDirectionType):
        stim_direction_type = StimDirectionType(stim_direction_type)
        warnings.warn(f"stim_direction_type should be a StimDirectionType enum, not a string. Converting {stim_direction_type} to StimDirectionType.")
    if stim_direction_type == StimDirectionType.FIRST:
        desired_stim = numpy.zeros((latent_d, 1))
        desired_stim[0] = 1
    elif stim_direction_type == StimDirectionType.FIRST2:
        desired_stim = numpy.zeros((latent_d, 2))
        desired_stim[0] = 1
        desired_stim[1] = 1
    elif stim_direction_type == StimDirectionType.COL:
        desired_stim = numpy.zeros((latent_d, 1))
        desired_stim[rng.choice(latent_d), 0] = 1
    elif stim_direction_type == StimDirectionType.RANDOM:
        desired_stim = rng.normal(size=(latent_d, 1))
        desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
    elif stim_direction_type == StimDirectionType.RANDOM_POSITIVE:
        desired_stim_high_d = rng.normal(size=(full_d, 1))
        desired_stim_high_d = desired_stim_high_d / numpy.linalg.norm(desired_stim_high_d)
        desired_stim_high_d = numpy.abs(desired_stim_high_d)
        desired_stim = u_to_latent_s(desired_stim_high_d)
        desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
    elif stim_direction_type == StimDirectionType.RANDOM_FEASIBLE:
        desired_stim_high_d = rng.normal(size=(full_d, 1))
        desired_stim_high_d = desired_stim_high_d / numpy.linalg.norm(desired_stim_high_d)
        desired_stim_high_d = numpy.abs(desired_stim_high_d).flatten()
        while (desired_stim_high_d > 0).sum() > max_l0_norm:
            desired_stim_high_d[rng.choice(len(desired_stim_high_d))] = 0
        desired_stim = u_to_latent_s(desired_stim_high_d)
        desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        desired_stim = desired_stim.reshape([-1,1])
    elif stim_direction_type == StimDirectionType.ONES:
        desired_stim_high_d = numpy.ones((full_d, 1))
        desired_stim = u_to_latent_s(desired_stim_high_d)
        desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
    elif stim_direction_type == StimDirectionType.NEG_ONES:
        desired_stim_high_d = -numpy.ones((full_d, 1))
        desired_stim = u_to_latent_s(desired_stim_high_d)
        desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
    else:
        raise ValueError(stim_direction_type)
    return desired_stim


def _hz_to_isi(x):
    return 1/x

class StimTimer:
    def __init__(
            self,
            inter_stim_interval_generator=None,
            stim_timing_method='regular',
            initial_nostim_period=1.,
    ):
        self.stim_timing_method = stim_timing_method
        self.initial_nostim_period = initial_nostim_period
        if inter_stim_interval_generator is None:
            inter_stim_interval_generator = cycle([1])
        self.inter_stim_interval_generator = inter_stim_interval_generator
        self.last_stim_time = None
        self.current_isi = None

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
            return kwargs['stim_time_rng'].random() < 1 / next(self.inter_stim_interval_generator) * kwargs[
                'input_array_dt']
        else:
            raise ValueError()



def sim_stim_design_stim(stim_designer: StimDesigner, sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t, u_to_s_model_type: StimResponseModelType):
    optimization_method = stim_designer.optimization_method
    if sr.stim_reg.n_observed <= stim_designer.n_random_initialization and (u_to_s_model_type == StimResponseModelType.KERNEL_REGRESSED or optimization_method == OptimizationMethod.PREV_SEEN):
        # u_to_s_model_type = 'identity'
        u_to_s_model_type = None
        optimization_method = OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS


    if optimization_method in {OptimizationMethod.JAXOPT, OptimizationMethod.JAXOPT_UNCONSTRAINED, OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED, OptimizationMethod.JAXOPT_SPARSE_CONSTRAINED, OptimizationMethod.PREV_SEEN}:
        stim_reg = sr.stim_reg
        previous_us = stim_reg.input_histories[1][:stim_reg.n_observed] if stim_reg.input_histories is not None else None
        if u_to_s_model_type == StimResponseModelType.KERNEL_REGRESSED:
            f = stim_reg.make_jax_pred_f()
            pred = sr.autoreg.predict(n_steps=0)
            u_to_s_function = Partial(_kernel_reg_u_to_s, stim_magnitude=stim_magnitude, f=f, pred=pred, current_t=current_t)
            designed_stim = stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
        elif u_to_s_model_type == StimResponseModelType.IDENTITY:
            u_to_s_function = Partial(_linear_u_to_s, A=equivalent_projection_matrix.T, stim_magnitude=stim_magnitude)
            designed_stim = stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
    elif optimization_method == OptimizationMethod.CHEAT_LOWD_VEC and u_to_s_model_type == StimResponseModelType.IDENTITY:
        designed_stim = stim_designer.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix)
    elif optimization_method in {OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS, OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS}:
        designed_stim = stim_designer.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix, optimization_method=optimization_method)
    else:
        raise ValueError()

    stim_designer.add_to_last_log({
        'time_of_stim': current_t,
        'equiv_proj_mat': equivalent_projection_matrix,
        # 'stim_reg': copy.deepcopy(stim_reg),
    })

    if (designed_stim == 0).all():
        designed_stim[0] = 1e-10
        warnings.warn("Stimulus was all zero!")  # TODO: handle this better

    return designed_stim



@dataclass(frozen=True)
class SimStimConfig:
    autoreg: type = StreamingKalmanFilter
    isi_generator: object = None
    exit_time: float = 60
    decay_rate: float = .8
    prosvd_k: int = 10
    stim_magnitude: float = 10
    max_l0_norm: int = 30
    attempt_correction: bool = True
    heed_stimuli: bool = True
    stim_time_delay: int = 0
    regressor_stim_delay: int = 0
    optimization_method: OptimizationMethod = OptimizationMethod.JAXOPT
    u_to_s_model_type: StimResponseModelType = StimResponseModelType.IDENTITY
    design_type: str = None  # TODO: currently unused, kept for parity with the old signature
    true_S: StimResponseType = StimResponseType.IDENTITY
    stim_timing_method: str = 'random'
    n_identity_prior: int = 10
    stim_direction_type: StimDirectionType = StimDirectionType.FIRST
    initial_nostim_period: float = 5
    stim_reg_maxlen: int = 500
    smoothing_tau: float = None
    centerer_init_size: int = 0
    last_dim_red: str = 'prosvd'
    show_tqdm: bool = False
    beh_decay_rate: float = .8
    v_design_use_full_u_s_map: bool = False
    delay_switch_time: float = None
    delay_switch_amount: int = 0


class SimulationResult(NamedTuple):
    sr: StimRegressor
    stim_designer: StimDesigner
    log: dict
    config: SimStimConfig

@save_to_cache('run_sim_stim', location='/mnt/data/gould_2026_cache/')
def run_sim_stim(
        input_array,
        rng,
        autoreg=StreamingKalmanFilter,
        stim_rate=1, # TODO: refactor out
        regular_stim_iter=None,  # TODO: refactor out
        isi_generator=None,
        exit_time=60,
        decay_rate=.8,
        prosvd_k=10,
        stim_magnitude=10,
        max_l0_norm=30,
        attempt_correction=True,
        heed_stimuli=True,
        stim_time_delay=0,
        regressor_stim_delay=0,
        design_method=None, # TODO: refactor out
        optimization_method=OptimizationMethod.JAXOPT,
        u_to_s_model_type=StimResponseModelType.IDENTITY,
        design_type=None,
        true_S=StimResponseType.IDENTITY,
        stim_timing_method='random',
        n_identity_prior=10,
        stim_direction_type=StimDirectionType.FIRST,
        initial_nostim_period=5,
        stim_reg_maxlen=500,
        smoothing_tau=None,
        centerer_init_size=0,
        last_dim_red='prosvd',
        show_tqdm=False,
        behavioral_data=ArrayWithTime(np.zeros((2,1)), [np.inf, np.inf]) * np.nan,
        beh_decay_rate=.8,
        v_design_use_full_u_s_map=False,
        delay_switch_time=None,
        delay_switch_amount=0,
):
    _init_time = time.perf_counter()
    timing_log = SimpleNamespace()
    timing_log.init_time = _init_time
    timing_log.loop_time = 0
    timing_log.stim_design = []
    timing_log.dimension_reduction = []
    timing_log.sr_update = []
    timing_log.per_loop = []
    timing_log.stim_reg_updated = []
    timing_log.in_sim_time = []


    assert (regular_stim_iter is not None) + (stim_rate is not None) + (isi_generator is not None) == 1
    if stim_rate:
        isi_generator = cycle([1/stim_rate])
    elif regular_stim_iter:
        isi_generator = map(_hz_to_isi, regular_stim_iter)
        assert stim_timing_method == 'regular'
        stim_timing_method = 'isi'
    del regular_stim_iter, stim_rate

    _optimization_method, _u_to_s_model_type = {
        'optimized learned u_to_s': (OptimizationMethod.JAXOPT, StimResponseModelType.KERNEL_REGRESSED),
        'optimized identity u_to_s': (OptimizationMethod.JAXOPT, StimResponseModelType.IDENTITY),
        'direct cheating': (OptimizationMethod.CHEAT_LOWD_VEC, StimResponseModelType.IDENTITY),
        'single neurons': (OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS, None),
        'many neurons': (OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS, None),
        None: (optimization_method, u_to_s_model_type),
    }[design_method]
    # single neurons
    # many neurons
    del design_method
    if optimization_method is not None:
        assert optimization_method == _optimization_method
    if _u_to_s_model_type is not None:
        assert u_to_s_model_type == _u_to_s_model_type

    config = SimStimConfig(
        autoreg=autoreg,
        isi_generator=isi_generator,
        exit_time=exit_time,
        decay_rate=decay_rate,
        prosvd_k=prosvd_k,
        stim_magnitude=stim_magnitude,
        max_l0_norm=max_l0_norm,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_time_delay=stim_time_delay,
        regressor_stim_delay=regressor_stim_delay,
        optimization_method=_optimization_method,
        u_to_s_model_type=_u_to_s_model_type,
        design_type=design_type,
        true_S=true_S,
        stim_timing_method=stim_timing_method,
        n_identity_prior=n_identity_prior,
        stim_direction_type=stim_direction_type,
        initial_nostim_period=initial_nostim_period,
        stim_reg_maxlen=stim_reg_maxlen,
        smoothing_tau=smoothing_tau,
        centerer_init_size=centerer_init_size,
        last_dim_red=last_dim_red,
        show_tqdm=show_tqdm,
        beh_decay_rate=beh_decay_rate,
        v_design_use_full_u_s_map=v_design_use_full_u_s_map,
        delay_switch_time=delay_switch_time,
        delay_switch_amount=delay_switch_amount,
    )
    del (autoreg, isi_generator, exit_time, decay_rate, prosvd_k, stim_magnitude, max_l0_norm,
         attempt_correction, heed_stimuli, stim_time_delay, regressor_stim_delay, optimization_method,
         u_to_s_model_type, design_type, true_S, stim_timing_method, n_identity_prior, stim_direction_type,
         initial_nostim_period, stim_reg_maxlen, smoothing_tau, centerer_init_size, last_dim_red, show_tqdm,
         beh_decay_rate, v_design_use_full_u_s_map, delay_switch_time, delay_switch_amount,
         _optimization_method, _u_to_s_model_type)

    stim_time_rng, other_rng = rng.spawn(2)


    sr = StimRegressor(
        autoreg=config.autoreg(),
        stim_reg=KernelRegressor(length_scales=[0.04, 0.04, 0.04], maxlen=config.stim_reg_maxlen),
        log_level=2,
        check_dt=True,
        attempt_correction=config.attempt_correction,
        heed_stimuli=config.heed_stimuli,
        stim_delay=config.regressor_stim_delay,
    )
    # NOTE: the old zong_stim.py pipeline overrode the default autoregressive residual-correction here with
    # `sr.stim_autoreg = StimAutoReg(n_steps_to_consider=6)` (default is `n_steps_to_consider=0`, i.e. off).
    # To restore that behavior, uncomment the next two lines (and `from .stim_regressor import StimAutoReg`),
    # or add a `stim_autoreg_n_steps=0` parameter to this function and use it here.
    # from .stim_regressor import StimAutoReg
    # sr.stim_autoreg = StimAutoReg(n_steps_to_consider=6)
    stim_designer = StimDesigner(
        max_l0_norm=config.max_l0_norm,
        rng_seed=other_rng.integers(2 ** 32),
        should_log=True,
        optimization_method=config.optimization_method, # todo:fix
        n_random_initialization=config.n_identity_prior
    )

    stim_timer = StimTimer(
        initial_nostim_period=config.initial_nostim_period,
        stim_timing_method=config.stim_timing_method,
        inter_stim_interval_generator=config.isi_generator,
    )

    sim_stim_calculator = SimulatedStimResponseCalculator(
        true_S=config.true_S,
        rng=other_rng,
    )
    sim_stim_adder = SimulatedStimAdder(
        stim_time_delay=config.stim_time_delay,
        decay=config.decay_rate
    )

    beh_sim_stim_adder = SimulatedStimAdder(
        stim_time_delay=config.stim_time_delay,
        decay=config.beh_decay_rate
    )

    log = {}


    centerer = CenteringEstimator(init_size=config.centerer_init_size, nan_when_uninitialized=True)
    if config.smoothing_tau is not None:
        smoother = KernelSmoother(tau=config.smoothing_tau/input_array.dt)
    else:
        smoother = Pipeline()

    pro = proSVD(k=config.prosvd_k)
    if config.last_dim_red == 'prosvd':
        last_dim_red_object = None
    elif config.last_dim_red == 'sjpca':
        last_dim_red_object = sjPCA()
    elif config.last_dim_red == 'mmica':
        last_dim_red_object = mmICA()
    else:
        raise ValueError()

    decided_stims = []
    stims = []
    latents = []
    behavior = []
    high_d_without_stim = []
    high_d_with_stim = []
    high_d_stims = []

    pbar = nullcontext()
    if config.show_tqdm:
        pbar = tqdm(total=min(round(input_array.t[-1]), config.exit_time))

    timing_log.init_time = time.perf_counter() - timing_log.init_time
    timing_log.loop_time = time.perf_counter()
    delay_switched = False
    with pbar:
        for data, stream in Pipeline().streaming_run_on([(input_array, 'neural_data'), (behavioral_data, 'behavioral_data')], return_output_stream=True):
            if stream == 'neural_data':
                timing_log.in_sim_time.append(data.t)
                timing_log.per_loop.append(time.perf_counter())
                timing_log.stim_design.append(time.perf_counter())

                # Simulates a change (partway through the run) in how many samples it takes for a stim to
                # affect the recorded signal / be corrected for. Used e.g. by the zong_stim figure.
                if config.delay_switch_time is not None and data.t > config.delay_switch_time and not delay_switched:
                    sim_stim_adder.stim_delay_queue = deque([0] * config.delay_switch_amount)
                    sr.stim_delay = sr.stim_delay + sr.dt * config.delay_switch_amount
                    delay_switched = True

                stim_decision = stim_timer.decide_whether_to_stim(data.t, stim_time_rng=stim_time_rng, input_array_dt=input_array.dt)
                decided_stims.append(ArrayWithTime(stim_decision, data.t))

                equivalent_projection_matrix = calculate_equivalent_projection_matrix(pro, last_dim_red_object)
                if stim_decision and equivalent_projection_matrix is not None:
                    if config.v_design_use_full_u_s_map:
                        u_to_latent_s = functools.partial(sim_stim_calculator.true_stim_result_in_latent_space, equivalent_projection_matrix=equivalent_projection_matrix)
                    else:
                        u_to_latent_s = lambda x: equivalent_projection_matrix.T @ x
                    desired_stim = desired_stim_direction(
                        latent_d=equivalent_projection_matrix.shape[1],
                        full_d=equivalent_projection_matrix.shape[0],
                        u_to_latent_s=u_to_latent_s,
                        stim_direction_type=config.stim_direction_type,
                        rng=other_rng,
                        max_l0_norm=stim_designer.max_l0_norm
                    )
                    designed_stim = sim_stim_design_stim(stim_designer, sr, config.stim_magnitude, desired_stim, equivalent_projection_matrix, current_t=data.t, u_to_s_model_type=config.u_to_s_model_type)
                    instantaneous_stim = designed_stim * config.stim_magnitude
                else:
                    instantaneous_stim = np.zeros(input_array.shape[1])
                timing_log.stim_design[-1] = time.perf_counter() - timing_log.stim_design[-1]

                stims.append(ArrayWithTime(instantaneous_stim, data.t))

                true_stim_result = sim_stim_calculator.true_stim_result(instantaneous_stim, equivalent_projection_matrix)

                sim_stim_adder.register_stim(true_stim_result)

                high_d_without_stim.append(data)
                pre_stim_data = data
                data = sim_stim_adder.run_for_X(data)
                # NOTE: the old zong_stim.py pipeline additionally smoothed the stim-injected data here via
                # `ss_adder_p = Pipeline([sim_stim_adder, KernelSmoother(tau=1)])` and stepped that pipeline
                # instead of calling `sim_stim_adder.run_for_X` directly. To restore, wrap `sim_stim_adder` in
                # a `Pipeline([sim_stim_adder, KernelSmoother(tau=1)])` above and `.step(data, stream='X')` it here.
                high_d_with_stim.append(data)
                high_d_stims.append(data - pre_stim_data)

                timing_log.dimension_reduction.append(time.perf_counter())
                data = centerer.step(data, stream='X')
                data = smoother.step(data, stream='X')
                data = pro.step(data, stream='X')
                if last_dim_red_object is not None:
                    data = last_dim_red_object.step(data, stream='X')
                timing_log.dimension_reduction[-1] = time.perf_counter() - timing_log.dimension_reduction[-1]
                latents.append(data)

                timing_log.stim_reg_updated.append(sr.stim_reg.n_observed)
                timing_log.sr_update.append(time.perf_counter())
                sr.step(ArrayWithTime(true_stim_result, data.t), stream='stim')
                stims_before_obs = set([stim.t for stim in sr.last_seen_stims])
                data = sr.step(data, stream='X')
                resolved_stim_ts = stims_before_obs - set([stim.t for stim in sr.last_seen_stims])
                timing_log.sr_update[-1] = time.perf_counter() - timing_log.sr_update[-1]
                timing_log.stim_reg_updated[-1] = timing_log.stim_reg_updated[-1] != sr.stim_reg.n_observed

                if config.heed_stimuli and len(resolved_stim_ts):
                    assert len(resolved_stim_ts) == 1
                    stim_t = list(resolved_stim_ts)[0]

                    obs = sr.stim_reg.get_obs(t=stim_t + sr.stim_delay)
                    d = {
                        'observed_s_hat':obs.pop('output'),
                        'observed_reg_input':[v for v in obs.values()]
                    }
                    stim_designer.add_to_last_log(d, assert_callback=lambda l: l['time_of_stim'] == stim_t)

                if config.show_tqdm:
                    pbar.update(round(float(data.t), 2) - pbar.n)

                timing_log.per_loop[-1] = time.perf_counter() - timing_log.per_loop[-1]
            elif stream == 'behavioral_data':
                def beh_S(point, bottom=-1.24, top=2.4):
                    point = point / 8
                    quadratic = point[0] ** 2 - 4 * point[1] ** 2
                    surface = np.tanh(quadratic) * (top - bottom) / 2
                    surface = surface - (-(top - bottom) / 2 - bottom)
                    if np.isnan(surface):
                        return 0
                    else:
                        return surface

                if data.t > 500 and len(behavior) % 1000 == 1:
                    true_beh_stim_result = beh_S(latents[-1][0])
                else:
                    true_beh_stim_result = 0


                beh_sim_stim_adder.register_stim(true_beh_stim_result)
                data = beh_sim_stim_adder.run_for_X(data)

                behavior.append(data)
            else:
                raise ValueError()

            if data.t > config.exit_time:
                break

    timing_log.loop_time = time.perf_counter() - timing_log.loop_time
    log['high_d_stims'] = ArrayWithTime.from_list(high_d_stims, squeeze_type='to_2d', drop_early_nans=True)
    log['high_d_without_stim'] = ArrayWithTime.from_list(high_d_without_stim, squeeze_type='to_2d', drop_early_nans=True)
    log['high_d_with_stim'] = ArrayWithTime.from_list(high_d_with_stim, squeeze_type='to_2d', drop_early_nans=True)
    assert np.allclose(log['high_d_with_stim'], log['high_d_stims'] + log['high_d_without_stim'])
    log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d', drop_early_nans=True)
    log['behavior'] = ArrayWithTime.from_list(behavior, squeeze_type='to_2d', drop_early_nans=True)
    if (log['high_d_stims'] == 0).all():
        warnings.warn("No stims delivered in sim-stim.")

    stim_intended_samples = ArrayWithTime.from_list(decided_stims, squeeze_type='to_2d')
    log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))
    stims = ArrayWithTime.from_list(stims, squeeze_type='to_2d', drop_early_nans=True)
    log['stims'] = stims.slice((stims != 0).any(axis=1))
    log['timing_log'] = timing_log

    sr.log['pred_error'] = ArrayWithTime.from_list(sr.log['pred_error'])



    return SimulationResult(sr=sr, stim_designer=stim_designer, log=log, config=config)
