import numpy as np

from gould_2026.datasets import Zong22Dataset
from gould_2026.estimator import ArrayWithTime

from gould_2026.prediction.kalman_filter import StreamingKalmanFilter
from sim_stim import make_srs, make_slices_tensor
import functools
from itertools import cycle
import matplotlib.pyplot as plt

zero_thresh = 0.05  # NOTE: only used for plot_1's visualization threshold now; see gould_2026/sim_stim.py
                     # for how to restore actual stim-value zeroing during simulation.
amount_to_add = 4
switch_time = 304
colors = ['#ca1469ff','#4d4d4dff']


def make_srs_zong(data, rng, n_runs=1, show_tqdm=False, overrides=None):
    if overrides is None:
        overrides = {}

    common = dict(
        stim_magnitude=10,
        design_method='optimized identity u_to_s',
        exit_time=np.inf,
        stim_rate=None,
        smoothing_tau=1,
        centerer_init_size=8 * 25,
        initial_nostim_period=30,
        regular_stim_iter=cycle([1 / 10, 1 / 3]),
        stim_timing_method='regular',
        autoreg=functools.partial(StreamingKalmanFilter, steps_between_refits=5),
        # NOTE: the mid-run delay change (used to be hardcoded in this file's loop) is now a first-class,
        # cache-friendly parameter of `run_sim_stim`; wired in here via `overrides` from `main()`.
    )

    to_run = {
        'learning from stim': common | dict(attempt_correction=True, heed_stimuli=True),
        # 'ignoring stim': common | dict(attempt_correction=False, heed_stimuli=True),
        'unaware of stim': common | dict(attempt_correction=False, heed_stimuli=False),
    }

    return make_srs(data, rng, to_run, n_runs=n_runs, show_tqdm=show_tqdm, overrides=overrides)


def plot_1(srs):
    i = 40
    sr = srs['learning from stim'][0]

    fig_1, axs = plt.subplots(ncols=2, figsize=(10,4), sharex=False, sharey=False, layout='constrained')

    latents = sr.log['latents'].slice_by_time(slice(30,None))
    axs[0].plot(latents[:, 0], latents[:, 1], alpha=.1, color='k')
    stim_s = sr.log['stim_intended_samples'].t - latents.dt

    l = 1
    r = 4.7
    ax_n = 0
    center_t = sr.log['stim_intended_samples'].t[i]
    latents = sr.log['latents'].slice_by_time(slice(center_t-l,center_t+r))
    line = axs[ax_n].plot(latents[:, 0], latents[:, 1], color='k', lw=3)
    stim_s = sr.log['stim_intended_samples'].slice_by_time(slice(center_t-l,center_t+r)).t - latents.dt
    latents_s = latents.slice_by_time(stim_s).reshape((-1, latents.shape[1]))
    axs[ax_n].plot(latents_s[:, 0], latents_s[:, 1], '.', color='r')

    for arrow_index in [17, 50]:
        axs[0].annotate('',
                        xytext=(latents[arrow_index, 0], latents[arrow_index, 1]),
                        xy=(latents[arrow_index+1, 0], latents[arrow_index+1, 1]),
                        arrowprops=dict(arrowstyle="simple", color='C0'),
                        size=11
                        )


    u = sr.stim_designer.log[i]['u']
    idx = np.argsort(np.abs(u))[::-1]
    # n_nonzero = np.linalg.norm(u,ord=0)
    n_nonzero = (np.abs(u) > zero_thresh).sum() # these were actually zeroed out with a custom line, this isn't a threshold
    print(f'{n_nonzero=}')

    high_d = sr.log['high_d_with_stim'].slice_by_time(slice(center_t-l,center_t+r))
    axs[1].plot(high_d.t, high_d[:,idx[:int(n_nonzero)]], color='k', lw=1)
    axs[1].set_xticks([302, 304, 306,308])
    for stim_t in stim_s:
        axs[1].axvline(stim_t, color='r')
    return fig_1




def plot_onestep_pred_error_decreasing(srs, row_info, make_slices_tensor):
    fig, axs = plt.subplots(nrows=len(row_info), layout='tight', figsize=(8, 2*len(row_info)+1), sharex=True, sharey=True)

    def p(ax, time_slice_type, space_slice_type, xlabel='time', sr_kind_keys=None, title=None, time_slice=None, last_half_average=False):
        if time_slice is  None:
            time_slice = slice(None, None)

        if sr_kind_keys is None:
            sr_kind_keys = srs.keys()
        for idx, sr_kind_key in reversed(list(enumerate(sr_kind_keys))):
            all_to_plot = []
            for sr in srs[sr_kind_key]:
                run_to_plot = make_slices_tensor(sr)
                sub_to_plot = run_to_plot[time_slice_type][space_slice_type].slice_by_time(time_slice)
                sub_to_plot = ArrayWithTime(np.linalg.norm(sub_to_plot, axis=1), sub_to_plot.t)
                all_to_plot.append(sub_to_plot)
            to_plot = ArrayWithTime(np.hstack(all_to_plot), np.hstack([p.t for p in all_to_plot])) # TODO: sort by time
            # TODO: you could do smoothing here
            ax.plot(to_plot.t, to_plot, '.-', color=f'C{idx}', label=sr_kind_key)
            if last_half_average:
                halfway = (to_plot.t.max() + to_plot.t.min()) / 2
                mean = float(to_plot.slice_by_time(slice(halfway, None)).mean())
                ax.axhline(mean, linestyle='--', color=f'C{idx}')
        # ax.legend(loc='upper right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('error norm')
        if title is None:
            title = f"time:'{time_slice_type}' space:'{space_slice_type}' norm error"
        ax.set_title(title)

    for idx, values in enumerate(row_info):
        p(ax=axs[idx], **values)

    return fig

def plot_2(srs):
    fig, axs = plt.subplots(nrows=1, figsize=np.array((9,6)), squeeze=False, layout='constrained', sharex=False, sharey=True)

    ax = axs[0,0]
    error = srs['unaware of stim'][0].log['pred_error']
    norm_error = np.linalg.norm(error, axis=(1,2))
    ax.plot(error.t, norm_error, '.-', color=colors[1], label='unaware of stim')

    error = srs['learning from stim'][0].log['pred_error']
    norm_error = np.linalg.norm(error, axis=(1,2))
    ax.plot(error.t, norm_error, '.-', color=colors[0], label='learning from stim')

    sr = srs['unaware of stim'][0]
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']
    stim_intended_samples.t[stim_intended_samples.t > switch_time] += amount_to_add * error.dt
    sliced_error, _ = ArrayWithTime.align_indices(error, stim_intended_samples)
    bin_slice = np.array([int(t in sliced_error.t) for t in error.t])
    bin_slice = np.convolve(bin_slice, np.array([0,0,0,0,1,1,1,1,1,1]), mode='same').astype(bool)
    sliced_error = error.slice(bin_slice)
    error_norms = np.linalg.norm(sliced_error, axis=(1,2))
    axs[0,0].axhline(np.nanmean(error_norms), linestyle='--', color=colors[1])
    print(f'unaware mean stim-centered error:  {np.nanmean(error_norms):.3f}')

    sr = srs['learning from stim'][0]
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']
    stim_intended_samples.t[stim_intended_samples.t > switch_time] += amount_to_add * error.dt
    sliced_error, _ = ArrayWithTime.align_indices(error, stim_intended_samples)
    bin_slice = np.array([int(t in sliced_error.t) for t in error.t])
    bin_slice = np.convolve(bin_slice, np.array([0,0,0,0,1,1,1,1,1,1]), mode='same').astype(bool)
    sliced_error = error.slice(bin_slice)
    error_norms = np.linalg.norm(sliced_error, axis=(1,2))
    axs[0,0].axhline(np.nanmean(error_norms), linestyle='--', color=colors[0])

    ax.set_ylim(0, .8)
    ax.axvline(switch_time, linestyle='--', color='gray')

    return fig

def main():
    def f():
        rng = np.random.default_rng(0)
        d = Zong22Dataset()
        data = d.neural_data

        srs = make_srs_zong(
            data, rng, n_runs=1, show_tqdm=True,
            overrides=dict(
                stim_magnitude=9.85,
                regressor_stim_delay=0 * data.dt,
                delay_switch_time=switch_time,
                delay_switch_amount=amount_to_add,
            ),
        )
        return srs

    srs = f()


    fig_1 = plot_1(srs)
    fig_2 = plot_2(srs)

    return fig_1, fig_2


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig_1, fig_2 = main()

    fig_1.savefig(args.output, bbox_inches="tight")
    fig_2.savefig(args.output.with_stem('zong_1step'), bbox_inches="tight")
