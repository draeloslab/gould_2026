import numpy as np

import json
import seaborn as sns
from sim_stim import make_srs, get_sim_stim_preset
from gould_2026.estimator import ArrayWithTime
from gould_2026.datasets import Zong22Dataset, Odoherty21Dataset, LDS
from gould_2026.prediction.vjf import VJF
from gould_2026.prediction.bubblewrap import Bubblewrap
from gould_2026.prediction.kalman_filter import StreamingKalmanFilter
from matplotlib.path import Path
import matplotlib.pyplot as plt
import pandas
from gould_2026.stim_designer import OptimizationMethod
from scipy.stats import linregress
from gould_2026.plotting import Palette

_vh = .5
verts = [ (-1., -_vh), (-1., _vh), (1., _vh), (1., -_vh), (-1., -_vh), ]
codes = [ Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
white_bar_path = Path(verts, codes)
violinplot_inner_kws = {'marker': white_bar_path, 'markersize': 3, 'markerfacecolor': 'white', }

def add_info_to_json(line_info):
    try:
        with open('output/collected_info.json', 'r+') as f:
            info = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        info = []

    info.append(line_info)
    with open('output/collected_info.json', 'w+') as f:
        json.dump(info, f)


def proportion_in_space(desired, designed):
    assert np.allclose(desired.T @ desired, np.eye(desired.shape[1]))
    proj = desired @ desired.T @ designed
    in_norm = np.linalg.norm(proj)
    total_norm = np.linalg.norm(designed)
    if total_norm == 0:
        ratio = 0
    else:
        ratio = in_norm / total_norm
    return ratio

def make_unit(x):
    x = np.squeeze(x)
    assert len(x.shape) == 1
    return x / np.linalg.norm(x)

def angle(a,b):
    return np.acos(make_unit(a) @ make_unit(b).flatten()) * 180/np.pi

def srs_to_l_df(srs):
    records = []
    for k, sr_list in srs.items():
        for sr_i, sr in enumerate(sr_list):
            latents: ArrayWithTime = sr.log['latents']
            for l_i, l in enumerate(sr.stim_designer.log):
                t_of_stim = l['time_of_stim']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                l['old_v'] = old_v
                l['this_v'] = this_v

                records.append(dict(sr_key=k, sr_i=sr_i, l_i=l_i, l=l))
    return pandas.DataFrame(records)


def extract_metrics(srs, preq_cutoff=None, metric_functions=None):
    if metric_functions is None:
        metric_functions = {
            'proportions': lambda l: proportion_in_space(l['v'], l['s']),
            'preq_errors': lambda l: np.linalg.norm(l['observed_s_hat'] - l['stim_reg'].predict(l['observed_reg_input'])) if l['stim_reg'] is not None else np.nan,
            'v_delta_errors': lambda l: proportion_in_space(l['v'], l['observed_s_hat']),
            's_delta_errors': lambda l: np.linalg.norm(l['s'] - l['observed_s_hat']),
            'angles': lambda l: angle(l['observed_s_hat'], l['v']),
            'mags_along': lambda l: l['observed_s_hat'] @ make_unit(l['v']),
            'mags': lambda l: np.linalg.norm(l['observed_s_hat']),
            'alignment_with_old_v': lambda l: angle(l['this_v'], l['old_v']),
            'v_mag_ratio': lambda l: np.linalg.norm(l['this_v']) / np.linalg.norm(l['old_v']),
        }
    metrics = {name: [] for name in metric_functions}

    for k, sr_list in srs.items():
        for m in metrics.values():
            m.append([])

        for sr in sr_list:
            for m in metrics.values():
                m[-1].append([])

            latents: ArrayWithTime = sr.log['latents']

            for l in sr.stim_designer.log:
                t_of_stim = l['time_of_stim']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                l['old_v'] = old_v
                l['this_v'] = this_v

                for name, m in metrics.items():
                    m[-1][-1].append(metric_functions[name](l))


            if preq_cutoff is not None:
                for m in metrics.values():
                    m[-1][-1] = m[-1][-1][:preq_cutoff]

    if preq_cutoff is None:
        preq_cutoff = np.inf
        for a in list(metrics.values())[0]:
            for b in a:
                if len(b) < preq_cutoff:
                    preq_cutoff = len(b)

        for k in metrics:
            metrics[k] = [[b[:preq_cutoff] for b in a] for a in metrics[k]]

    return metrics

def apply_lambda(srs, f, preq_cutoff=None):
    return extract_metrics(srs, preq_cutoff=preq_cutoff, metric_functions={'custom': f})['custom']

def unpack_metrics(metrics):
    if isinstance(metrics, dict):
        return metrics['proportions'], metrics['preq_errors'], metrics['v_delta_errors'], metrics['s_delta_errors'], metrics['angles'], metrics['mags_along'], metrics['mags'], metrics['alignment_with_old_v'], metrics['v_mag_ratio']
    else:
        return metrics


def open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=True, legend=False):
    fig, axs = plt.subplots(ncols=2, nrows=1, squeeze=False, layout='constrained', figsize=(2*4, 1*4))

    ax: plt.Axes = axs[0,0]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1)
    for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
        trendline = np.mean(errors, axis=0)
        ax.plot(trendline, color=f'C{i}', lw=1.5, label=f'{k} {trendline[trendline.size//2:].mean():.2f} ')
    ax.set_title('$s_{\\text{obs}}$ along $v$')

    ax: plt.Axes = axs[0,1]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1, )

    for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
        trendline = np.mean(errors, axis=0)

        line_last_half = trendline[trendline.size//2:]
        line_info = dict(dataset=args.dataset, type_of_dim_red=args.type_of_dim_red, type_of_autoreg=args.type_of_autoreg, condition=k, metric='s_hat', mean=line_last_half.mean(), std=line_last_half.std())

        add_info_to_json(line_info)

        ax.plot(trendline, color=f'C{i}', lw=1.5, label=f'{k} {line_info["mean"]:.2f} +/- {line_info["std"]:.2f}')
    ax.set_title('$\\Vert \\hat s_{obs} - \\hat S_{i-1}(x_i, u_i, t_i) \\Vert$')

    if legend:
        for ax in axs.flatten():
            ax.legend()


    return fig

N = 2

def compare_opt_by_target(closed=False, optimization_method=OptimizationMethod.JAXOPT):

    from gould_2026.save_to_cache import save_to_cache
    @save_to_cache('compare_opt_by_target', location='/mnt/data/gould_2026_cache/')

    def to_cache(n_runs=N, closed=closed, optimization_method=optimization_method):
        d = Odoherty21Dataset()
        data = d.neural_data

        u_to_s_model_type = 'identity' if not closed else 'kernel_regressed'
        common = dict(stim_rate=1 / 2, stim_magnitude=10, exit_time=130)
        to_run = {}
        stim_direction_types = ('random_feasible', 'first', 'ones', 'random+', 'col', 'random', '-ones')
        for stim_direction_type in stim_direction_types:
            inner_common = common | dict(stim_direction_type=stim_direction_type)
            to_run.update({
                f'normal {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=optimization_method, u_to_s_model_type=u_to_s_model_type, ),
                f'shuffled {stim_direction_type}': inner_common | dict(true_S='high_d_permuted', optimization_method=optimization_method, u_to_s_model_type=u_to_s_model_type),
                f'many {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS, u_to_s_model_type=None),
                f'single {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS, u_to_s_model_type=None),
            })

        srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=n_runs, show_tqdm=True)
        return srs

    srs = to_cache(n_runs=N)

    l_df = srs_to_l_df(srs)
    l_df[['optim_method', 'stim_direction_type']] = l_df['sr_key'].str.split(' ', expand=True)

    for stim_direction_type_to_drop in [
        'random+',
        'col'
    ]:
        l_df.drop(index=l_df.index[(l_df.stim_direction_type == stim_direction_type_to_drop)], inplace=True)


    order = ('-ones', 'ones', 'random', 'random_feasible', 'first')
    l_df.sort_values(by='stim_direction_type', inplace=True, key=lambda x: x.apply(order.index))

    stim_direction_type_subs = {'first': 'Q_0', 'random_feasible': 'feasible', '-ones': 'negative', 'ones': 'dense', 'random': 'random'}


    order = ('first', '-ones', 'ones', 'random', 'random_feasible')
    l_df.sort_values(by='stim_direction_type', inplace=True, key=lambda x: x.apply(order.index))
    l_df['stim_direction_type'] = l_df['stim_direction_type'].replace(stim_direction_type_subs)

    fig, axs= plt.subplots(ncols=1, nrows=1, figsize=(8,8), squeeze=False, layout='constrained')


    sub_df = l_df[(l_df['optim_method'].apply(lambda x: x in ['normal', 'many']))]
    ax: plt.Axes = axs[0, 0]
    metric_name = 'angles(s_obs,v)'
    sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))

    stim_direction_types = sub_df['stim_direction_type'].unique()
    palette = {'normal': '#00000000', 'many': 'gray'}
    sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, hue='optim_method', orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws, palette=palette, order=stim_direction_types)


    for i, collection in enumerate(ax.collections):
        if hasattr(collection, 'get_facecolor'):
            if i % 2 == 0:
                collection.set_facecolor(Palette[stim_direction_types[i//2]])

    return fig, []

def plot_optim_open_vs_closed(args):
    def _f(n_runs=N, args_dataset=args.dataset, args_type_of_dim_red=args.type_of_dim_red, args_type_of_autoreg=args.type_of_autoreg):
        if args_dataset == 'odoherty21':
            data = Odoherty21Dataset().neural_data
        elif args_dataset == 'zong22':
            data = Zong22Dataset().neural_data
        else:
            raise ValueError()
        
        if args_type_of_autoreg == 'kf':
            autoreg = StreamingKalmanFilter
        elif args_type_of_autoreg == 'bw':
            autoreg = Bubblewrap
        elif args_type_of_autoreg == 'vjf':
            autoreg = VJF

        to_run, _ = get_sim_stim_preset(comparison_preset='optim_open_vs_closed')
        srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=n_runs, show_tqdm=True, overrides=dict(last_dim_red=args_type_of_dim_red, autoreg=autoreg))
        return srs

    from gould_2026.save_to_cache import save_to_cache
    f = save_to_cache('optim_open_vs_closed', location='/mnt/data/gould_2026_cache/')(_f)

    srs = f(n_runs=1, _recalculate_cache_value=True)

    proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(
        extract_metrics(srs, preq_cutoff=None))
    fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=True, legend=True)
    fig.axes[0].set_title(f'$s_{{\\text{{obs}}}}$ along $v$, {args.dataset} {args.type_of_dim_red} {args.type_of_autoreg}')
 
    l_df = srs_to_l_df(srs)
    fig2, axs = plt.subplots(ncols=2, squeeze=False, figsize=(10,4), layout='constrained')
    l_df[['open_closed', 'true_s']] = l_df['sr_key'].str.split(' ', expand=True)

    l_df['angle(s_obs,v)'] = l_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
    l_df['s_obs along v'] = l_df.l.apply(lambda l: proportion_in_space(l['v'], l['observed_s_hat']))


    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='angle(s_obs,v)', ax=axs[0,0], density_norm='width',inner_kws = violinplot_inner_kws) #  density_norm='width',
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='s_obs along v', ax=axs[0,1], density_norm='width',inner_kws = violinplot_inner_kws)

    fig3, axs = plt.subplots(figsize=(10, 8), squeeze=False, layout='constrained')
    errors = []
    for k, v in srs.items():
        errors.append([])
        for sr in v:
            e = ArrayWithTime.from_list(sr.log['pred_error'],squeeze_type='to_2d')
            errors[-1].append(ArrayWithTime(np.linalg.norm(e, axis=1), e.t))



    ax: plt.Axes = axs[0,0]
    for i, (k, es) in enumerate(zip(srs.keys(), errors)):
        for j, e in enumerate(es):
            ax.plot(e.t, e, color=f'C{i}', alpha=0.1, )

    for i, (k, es) in enumerate(zip(srs.keys(), errors)):
        trendline = np.mean(es, axis=0)
        line_last_half = trendline[trendline.size//2:]
        line_info = dict(dataset=args.dataset, type_of_dim_red=args.type_of_dim_red, type_of_autoreg=args.type_of_autoreg, condition=k, metric='1step_pred', mean=line_last_half.mean(), std=line_last_half.std())
        add_info_to_json(line_info)
        ax.plot(e.t, trendline, color=f'C{i}', lw=1.5, label=f'{k} {line_info["mean"]:.2f} +/- {line_info["std"]:.2f}')

    ax.set_title(f'{args.dataset} {args.type_of_dim_red} {args.type_of_autoreg} 1 step pred error')
    ax.legend()

    fig4, axs4 = plt.subplots(figsize=(8, 8), squeeze=False, layout='constrained')
    l_df['n_opt_iterations'] = l_df.l.apply(lambda d: len(d['intermediate_xs']) if 'intermediate_xs' in d else np.nan)
    sns.stripplot(data=l_df, x='sr_key', y='n_opt_iterations', ax=axs4[0,0])

    fig5, axs5 = plt.subplots(figsize=(5, 5), nrows=2, ncols=2, sharex=True, sharey=True, squeeze=False, layout='constrained')

    from gould_2026.utils import angle_between
    l_df['theta'] = l_df['l'].apply(lambda l: angle_between(l['v'], l['observed_s_hat'], radians=False))
    l_df['r'] = l_df['l'].apply(lambda l: np.linalg.norm(l['observed_s_hat']))
    l_df['t'] = l_df['l'].apply(lambda l: l['time_of_stim'])

    min_norm = 10
    max_angle = 20

    for k, ax in zip(l_df.sr_key.unique(), axs5.flatten()):
        sub_df = l_df[l_df.sr_key == k]
        sub_df = sub_df[sub_df.t > sub_df.t.median()]
        ax.scatter(sub_df['theta'], sub_df['r'], s=1, label=k, color='k')
        patch = plt.Rectangle(xy=(0,min_norm), width=max_angle, height=100, color='r', alpha=.1)
        ax.add_patch(patch)
        ax.set_xlim(xmin=0, xmax=180)
        ax.text(0.99, 0.97, k + f"\n {((sub_df.theta < max_angle) & (sub_df.r > min_norm)).sum()} / {len(sub_df)}", transform=ax.transAxes, ha='right', va='top')

        ax.set_ylim(0,45)
        ax.axvline(90, linestyle='--', color='gray', alpha=0.5)

    for ax in axs[-1,:]:
        ax.set_xlabel('Angle between v and s (degrees)')

    for ax in axs[:,0]:
        ax.set_ylabel('Norm of s')



    return fig, [fig2, fig3, fig4, fig5]

def plot_optim_open_vs_closed_toy():
    def f():
        n_revolutions = 80
        obs_d = 130

        rng = np.random.default_rng(4)

        all_srs = []
        for _ in range(N):
            lds = LDS.circular_lds(rng=rng, obs_d=obs_d)
            _, data, _ = lds.simulate(int(lds.transitions_per_rotation * n_revolutions), rng=rng, initial_state=np.array([20, 0]))
            t = np.arange(data.shape[0]) * 1 / lds.transitions_per_rotation
            data = ArrayWithTime(data, t)

            to_run, _ = get_sim_stim_preset(comparison_preset='optim_open_vs_closed_toy')
            srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=1, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))
            all_srs.append(srs)

        srs = {k: [sub_srs[k][0] for sub_srs in all_srs] for k in srs.keys()}
        return srs

    srs = f()

    proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(
        extract_metrics(srs, preq_cutoff=None))
    fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False)


    l_df = srs_to_l_df(srs)
    fig2, axs = plt.subplots(ncols=2, squeeze=False, layout='constrained')
    l_df[['open_closed', 'true_s']] = l_df['sr_key'].str.split(' ', expand=True)

    l_df['angle(s_obs,v)'] = l_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
    l_df['s_obs along v'] = l_df.l.apply(lambda l: proportion_in_space(l['v'], l['observed_s_hat']))
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='angle(s_obs,v)', ax=axs[0,0], width=1, density_norm='width',inner_kws = violinplot_inner_kws)
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='s_obs along v', ax=axs[0,1], width=1, density_norm='width',inner_kws = violinplot_inner_kws)

    return fig, [fig2]

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    parser.add_argument( "--type-of-dim-red", type=str, required=False)
    parser.add_argument( "--type-of-autoreg", type=str, required=False, default='kf')
    parser.add_argument( "--dataset", type=str, required=False, default='Odoherty21')
    parser.add_argument( "--closed-loop", required=False, action='store_true')
    parser.add_argument( "--optimization-method", type=OptimizationMethod, required=False, default=OptimizationMethod.JAXOPT, choices=[x.value for x in OptimizationMethod])
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    match args.type_of_plot:
        case 'optim_col_vs_rand':
            d = Odoherty21Dataset()
            data = d.neural_data

            to_run, _ = get_sim_stim_preset(comparison_preset='optim_col_vs_rand')
            srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=N, show_tqdm=True)


            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(extract_metrics(srs, preq_cutoff=50))

            fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(8,4), layout='constrained')
            to_plot = {k:v for k, v in zip(srs.keys(), [x[0] for x in proportions])}
            sns.violinplot(to_plot, orient='v', ax=axs[0,0], width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(to_plot, orient='v', ax=axs[0,0])

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                for j, e in enumerate(errors):
                    axs[0,1].plot(e, color=f'C{i}', alpha=0.1)
            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                trendline = np.mean(errors, axis=0)
                axs[0,1].plot(trendline, color=f'C{i}', lw=1.5)
            axs[0, 1].semilogy()


        case 'compare_opt_by_target':
            fig, extra_figs = compare_opt_by_target(closed=args.closed_loop, optimization_method=args.optimization_method)
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem +f'_extra_{i}'), bbox_inches="tight", transparent=True)

        case 'optim_open_vs_closed':
            fig, extra_figs = plot_optim_open_vs_closed(args)

            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem + f'_extra_{i}'), bbox_inches="tight", transparent=True)

        case 'optim_open_vs_closed_toy':
            fig, extra_figs = plot_optim_open_vs_closed_toy()
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem + f'_extra_{i}'), bbox_inches="tight", transparent=True)
        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight", transparent=True)
