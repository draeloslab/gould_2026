import matplotlib.pyplot as plt
from gould_2026.datasets import LDS
import numpy as np
from learn_s_hat_toy import n_rotations, stims_per_rotation, stim_magnitude, noise_variance
from gould_2026.plotting import Palette, paper_plot_context
from typing import Literal

def show_toy_dataset(plot_type: Literal['curvy', 'curvy_spins', 'curvy_flips']):

    rng = np.random.default_rng(0)

    def u_function(lds, state, i, rng):
        u = np.zeros(lds.B.shape[0])
        if i == 20:
            u[2] = 5
        return u

    stim_steps = {16, 32, 80, 150} # 52
    def true_S(lds, state, i, rng):
        u = np.zeros(3)
        if i in stim_steps:
            u[2] = stim_magnitude * state[0] / np.linalg.norm(state[:2])
        return u

    # show_toy_n_turns = log_for_tex(key='show_toy_n_turns', value=10, current_file=__file__, output_directory=args.output.parent)
    # TODO: depreciated
    show_toy_n_turns = 10

    _, Y, stim = LDS.run_nest_dynamical_system(show_toy_n_turns, stims_per_rotation=stims_per_rotation, stim_magnitude=stim_magnitude, rng=rng, u_function=true_S, noise=noise_variance, radius=15)

    if plot_type == 'curvy':
        pass
    elif plot_type == 'curvy_spins':
        theta = np.pi/180 * 70
        Y[:, :2] = Y[:, :2] @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif plot_type == 'curvy_flips':
        Y[:, 2] = -Y[:, 2]

    with paper_plot_context():
        fig1, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(1,1))
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2])


        ax.axis('equal')
        if plot_type != 'curvy_flips':
            ax.view_init(elev=24, azim=147, roll=0)
        else:
            ax.view_init(elev=-24, azim=147, roll=0)
        ax.axis((-18., 15., -15., 15., -10., 10.))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        # for i in range(3):
        #     start = np.array([-10,-15,7.5])
        #     delta = np.zeros(3)
        #     delta[i] = 5
        #     end = start + delta
        #     ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=f'C{i}')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_axis_off()

    fig2, ax = plt.subplots()

    ax.plot(Y[:,:2], color='k', alpha=.5)
    ax.plot(Y[:,2], color='k')

    for step in stim_steps:
        ax.axvline(step, color='red', alpha=.5, linestyle='--')

    return fig1, fig2

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--plot-type", type=str, required=True)
    args = parser.parse_args()

    fig1, fig2 = show_toy_dataset(args.plot_type)

    fig1.savefig(args.output, bbox_inches="tight", transparent=True)
    fig2.savefig(args.output.with_name(args.output.stem + "_2" + args.output.suffix), bbox_inches="tight", transparent=True)
