import jax
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
import json
import seaborn as sns
from sim_stim import make_srs, get_sim_stim_preset
from gould_2026.sim_stim import StimDirectionType, StimResponseType
from gould_2026.estimator import ArrayWithTime
from gould_2026.datasets import Zong22Dataset, Odoherty21Dataset, LDS
from gould_2026.prediction.vjf import VJF
from gould_2026.utils import angle_between
from gould_2026.prediction.bubblewrap import Bubblewrap
from gould_2026.prediction.kalman_filter import StreamingKalmanFilter
from matplotlib.path import Path
import matplotlib.pyplot as plt
import pandas
from gould_2026.stim_designer import OptimizationMethod
import scipy.stats
import io
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


def open_v_closed_plot(srs, show_individuals=True, legend=False):
    proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(extract_metrics(srs, preq_cutoff=None))

    # TODO: args is global here
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

        # line_last_half = trendline[trendline.size//2:]
        # line_info = dict(dataset=args.dataset, type_of_dim_red=args.type_of_dim_red, type_of_autoreg=args.type_of_autoreg, condition=k, metric='s_hat', mean=line_last_half.mean(), std=line_last_half.std())
        # add_info_to_json(line_info)

        ax.plot(trendline, color=f'C{i}', lw=1.5)
    ax.set_title('$\\Vert \\hat s_{obs} - \\hat S_{i-1}(x_i, u_i, t_i) \\Vert$')

    if legend:
        for ax in axs.flatten():
            ax.legend()


    return fig


from gould_2026.save_to_cache import save_to_cache


@save_to_cache('make_table_over_target_type', location='/mnt/data/gould_2026_cache/')
def make_table_over_target_type(n_runs, stim_direction_types):
    d = Odoherty21Dataset()
    data = d.neural_data

    common = dict(stim_rate=1 / 2, stim_magnitude=10, exit_time=130)
    to_run = {}
    for closed in [False, True]:
        u_to_s_model_type = 'identity' if not closed else 'kernel_regressed'
        for stim_direction_type in stim_direction_types:
            inner_common = common | dict(stim_direction_type=stim_direction_type)

            for optimization_method in [
                OptimizationMethod.JAXOPT,
                OptimizationMethod.JAXOPT_SPARSE_CONSTRAINED,
                OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED,
                OptimizationMethod.JAXOPT_UNCONSTRAINED,
                OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS,
            ]:
                to_run[f'{optimization_method} {stim_direction_type} {closed}'] = inner_common | dict(true_S=StimResponseType.IDENTITY, optimization_method=optimization_method, u_to_s_model_type=u_to_s_model_type)
    srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=n_runs, show_tqdm=True)
    l_df = srs_to_l_df(srs)

    l_df[['optim_method', 'stim_direction_type', 'closed']] = l_df['sr_key'].str.split(' ', expand=True)
    l_df['closed'] = l_df['closed'].map({'True': True, 'False': False})

    stim_direction_type_subs = {'first': 'Q_0', 'random_feasible': 'feasible', '-ones': 'negative', 'ones': 'dense', 'random': 'random'}
    l_df['display_stim_direction_type'] = l_df['stim_direction_type'].replace(stim_direction_type_subs)


    l_df['angles(s_obs,v)'] = l_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))

    return l_df


def compare_opt_by_target(closed=False, optimization_method=OptimizationMethod.JAXOPT, n_runs=10):
    stim_direction_types = ('random_feasible', 'first', 'ones', 'random', '-ones')

    l_df = make_table_over_target_type(n_runs=n_runs, stim_direction_types=stim_direction_types)

    order = ('Q_0', 'negative', 'dense', 'random', 'feasible')
    l_df.sort_values(by='display_stim_direction_type', inplace=True, key=lambda x: x.apply(order.index))

    sub_df = l_df[
        (l_df['closed'] == closed)
        &
        l_df['optim_method'].apply(lambda x: x in [optimization_method, OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS])
    ]

    sub_df['optim_method'] = sub_df['optim_method'].map({
        OptimizationMethod.JAXOPT: 'normal',
        OptimizationMethod.JAXOPT_UNCONSTRAINED: 'normal',
        OptimizationMethod.JAXOPT_SPARSE_CONSTRAINED: 'normal',
        OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED: 'normal',
        OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS: 'many',
    })



    fig, axs= plt.subplots(ncols=1, nrows=1, figsize=(8,8), squeeze=False, layout='constrained')


    ax: plt.Axes = axs[0, 0]
    metric_name = 'angles(s_obs,v)'
    palette = {'normal': '#00000000', 'many': 'gray'}
    sns.violinplot(sub_df, x='display_stim_direction_type', y=metric_name, hue='optim_method', orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws, palette=palette, order=order)


    # test output file
    test_result_file = io.StringIO()
    stim_direction_types = sub_df['display_stim_direction_type'].unique()
    for stim_direction_type in stim_direction_types:
        test_result_file.write(f"comparison for '{stim_direction_type}', optimized ('normal') vs random ('many') [.05, .5, .95]\n")

        to_compare = dict()
        for optim_method in sub_df['optim_method'].unique():
            x = sub_df[(sub_df['display_stim_direction_type'] == stim_direction_type) & (sub_df['optim_method'] == optim_method)][metric_name]
            a, b, c = np.quantile(x, [.05, .5, .95])
            test_result_file.write(f'{optim_method}: [{a:06.3f}, {b:06.3f}, {c:06.3f}]')
            threshold = .9
            centered_to_median = np.abs(x - np.quantile(x, .5))
            test_result_file.write(f' median to .9 = {float(np.quantile(centered_to_median, threshold)):06.3f}\n')
            to_compare[optim_method] = x

        test_result = scipy.stats.wilcoxon(to_compare['normal'], to_compare['many'])
        test_result_file.write(f'p = {test_result.pvalue}\n\n')


    x = sub_df[(sub_df['display_stim_direction_type'] == 'feasible') & (sub_df['optim_method'] == 'normal')][metric_name]
    y = sub_df[(sub_df['display_stim_direction_type'] == 'dense') & (sub_df['optim_method'] == 'normal')][metric_name]
    test_result = scipy.stats.wilcoxon(x, y)
    test_result_file.write(f'feasible vs dense:\n')
    test_result_file.write(f'p = {test_result.pvalue}\n\n')

    x = sub_df[(sub_df['display_stim_direction_type'] == 'feasible') & (sub_df['optim_method'] == 'normal')][metric_name]
    y = sub_df[(sub_df['display_stim_direction_type'] == 'negative') & (sub_df['optim_method'] == 'normal')][metric_name]
    test_result = scipy.stats.wilcoxon(x, y)
    test_result_file.write(f'feasible vs negative:\n')
    test_result_file.write(f'p = {test_result.pvalue}\n\n')



    for i, collection in enumerate(ax.collections):
        if hasattr(collection, 'get_facecolor'):
            if (collection.get_facecolor() == np.array([0,0,0,1])).all(): # in palette, 'normal' gets black
                collection.set_facecolor(Palette[order[i//len(palette)]])

    return fig, [], [test_result_file]

def cross_method_target_tests(closed=False, n_runs=10):
    stim_direction_types = ('random_feasible', 'first', 'ones', 'random', '-ones')

    l_df = make_table_over_target_type(n_runs=n_runs, stim_direction_types=stim_direction_types)

    test_result_file = io.StringIO()

    for target in ['feasible', 'random']:
        metric_name = 'angles(s_obs,v)'
        method1 = OptimizationMethod.JAXOPT
        method2 = OptimizationMethod.JAXOPT_POSITIVE_CONSTRAINED

        target_slice = l_df['display_stim_direction_type'] == target
        closed_slice = l_df['closed'] == closed

        method_slice = l_df['optim_method'] == method1
        standard_performance = l_df[target_slice & closed_slice & method_slice][metric_name]

        method_slice = l_df['optim_method'] == method2
        relaxed_performance = l_df[target_slice & closed_slice & method_slice][metric_name]

        test_result = scipy.stats.wilcoxon(standard_performance, relaxed_performance)
        test_result_file.write(f'{method1} vs {method2} performance ({target=} {closed=}, {metric_name=}):\n')
        test_result_file.write(f'{method1} median: {np.quantile(standard_performance, .5):.2f}\n')
        test_result_file.write(f'{method2} median: {np.quantile(relaxed_performance, .5):.2f}\n')
        test_result_file.write(f'difference: {np.quantile(standard_performance, .5) - np.quantile(relaxed_performance, .5):.2f}\n')
        test_result_file.write(f'p={test_result.pvalue}\n\n')


    fig, ax = plt.subplots()

    sns.stripplot(data=l_df[closed_slice], x='display_stim_direction_type', hue='optim_method', y=metric_name, ax=ax, dodge=True)

    return fig, test_result_file


def plot_optim_open_vs_closed_toy(n_runs=10):
    def f():
        n_revolutions = 80
        obs_d = 130

        rng = np.random.default_rng(4)

        all_srs = []
        for _ in range(n_runs):
            lds = LDS.circular_lds(rng=rng, obs_d=obs_d)
            _, data, _ = lds.simulate(int(lds.transitions_per_rotation * n_revolutions), rng=rng, initial_state=np.array([20, 0]))
            t = np.arange(data.shape[0]) * 1 / lds.transitions_per_rotation
            data = ArrayWithTime(data, t)

            to_run, _ = get_sim_stim_preset(comparison_preset='optim_open_vs_closed_toy')
            srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=n_runs, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))
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
    parser.add_argument( "--n-runs", type=int, required=True, default=10)
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
            srs = make_srs(data=data, rng=rng, to_run=to_run, n_runs=args.n_runs, show_tqdm=True)


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
            fig, extra_figs, text_files = compare_opt_by_target(closed=args.closed_loop, optimization_method=args.optimization_method, n_runs=args.n_runs)
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem +f'_extra_{i}'), bbox_inches="tight", transparent=True)
            for i, extra_text in enumerate(text_files):
                with args.output.with_stem(args.output.stem +f'_extra_{i}').with_suffix('.txt').open('w') as f:
                    print(extra_text.getvalue(), file=f)
        case 'cross_method_target_tests':
            fig, test_output = cross_method_target_tests(closed=args.closed_loop, n_runs=args.n_runs)
            with open(args.output.with_suffix('.txt'), 'w') as f:
                f.write(test_output.getvalue())
        case 'optim_open_vs_closed':
            raise NotImplementedError('This functionality was moved to the open_vs_closed notebook.')
        case 'optim_open_vs_closed_toy':
            fig, extra_figs = plot_optim_open_vs_closed_toy(n_runs=args.n_runs)
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem + f'_extra_{i}'), bbox_inches="tight", transparent=True)
        case _:
            raise ValueError()


    if fig is not None:
        fig.savefig(args.output, bbox_inches="tight", transparent=True)
