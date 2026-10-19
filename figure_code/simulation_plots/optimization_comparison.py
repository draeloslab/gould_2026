import jax
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
import seaborn as sns
from gould_2026.sim_stim import StimResponseType, SimulationResult, run_simulations, SimStimConfig
from gould_2026.estimator import ArrayWithTime
from gould_2026.datasets import Odoherty21Dataset
import pandas
from gould_2026.stim_designer import OptimizationMethod
from gould_2026.utils import angle_between




def simulations_to_l_df(simulations: dict[str, list[SimulationResult]]):
    records = []
    for k, result_list in simulations.items():
        for sr_i, result in enumerate(result_list):
            latents: ArrayWithTime = result.log['latents']
            for l_i, l in enumerate(result.stim_designer.log):
                t_of_stim = l['time_of_stim']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                l['old_v'] = old_v.as_array()
                l['this_v'] = this_v.as_array()

                records.append(dict(sr_key=k, sr_i=sr_i, l_i=l_i, l=l))
    return pandas.DataFrame(records)

def l_df_add_columns(l_df, open_closed_column=True, nan_missing_s_hat=True):
    if open_closed_column:
        l_df[['open_closed', 'true_s']] = l_df['sr_key'].str.split(' ', expand=True)
    l_df['angle(s,v)'] = l_df.l.apply(lambda l: angle_between(l['s'], l['v']))
    l_df['v'] = l_df.l.apply(lambda x: x['v'])
    l_df['u'] = l_df.l.apply(lambda x: x['u'])
    l_df['high_d_v'] = l_df.apply(lambda x: x['l']['equiv_proj_mat'] @ x['v'], axis=1)
    l_df['ones_theta'] = l_df['l'].apply(lambda l: angle_between(l['u'], l['u'] * 0 + 1, radians=False))
    l_df['t'] = l_df['l'].apply(lambda l: l['time_of_stim'])
    l_df['Q'] = l_df.l.apply(lambda x: x['equiv_proj_mat'])

    if nan_missing_s_hat:
        l_df['observed_s_hat'] = l_df.l.apply(lambda x: x['observed_s_hat'] if 'observed_s_hat' in x else np.nan * x['v'])

    l_df['r'] = l_df['observed_s_hat'].apply(np.linalg.norm)

    l_df['theta'] = l_df.apply(lambda r: angle_between(r['v'], r['observed_s_hat'], radians=False, subspace=False), axis=1)
    l_df['theta_space'] = l_df.apply(lambda r: angle_between(r['v'], r['observed_s_hat'], radians=False, subspace=True), axis=1)
    l_df['angle(s_obs,v)'] = l_df['theta']

    return l_df


from gould_2026.save_to_cache import save_to_cache


@save_to_cache('make_table_over_target_type', location='/mnt/data/gould_2026_cache/')
def make_table_over_target_type(n_runs, stim_direction_types, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    d = Odoherty21Dataset()
    data = d.neural_data

    common = SimStimConfig(isi_generator=[2], stim_magnitude=10, exit_time=130, stim_timing_method='isi')
    to_run = {}
    for closed in [False, True]:
        u_to_s_model_type = 'identity' if not closed else 'kernel_regressed'
        for stim_direction_type in stim_direction_types:
            inner_common = common.update(stim_direction_type=stim_direction_type)

            for optimization_method in [
                OptimizationMethod.LBFGS,
                OptimizationMethod.LBFGS_SPARSE_CONSTRAINED,
                OptimizationMethod.LBFGS_POSITIVE_CONSTRAINED,
                OptimizationMethod.LBFGS_UNCONSTRAINED,
                OptimizationMethod.RANDOM_MANY_NEURONS,
            ]:
                to_run[f'{optimization_method} {stim_direction_type} {closed}'] = inner_common.update(true_S=StimResponseType.IDENTITY, optimization_method=optimization_method, u_to_s_model_type=u_to_s_model_type)
    sims = run_simulations(data=data, rng=rng, to_run=to_run, n_runs=n_runs, show_tqdm=True)
    l_df = simulations_to_l_df(sims)

    l_df[['optim_method', 'stim_direction_type', 'closed']] = l_df['sr_key'].str.split(' ', expand=True)
    l_df['closed'] = l_df['closed'].map({'True': True, 'False': False})


    return l_df

