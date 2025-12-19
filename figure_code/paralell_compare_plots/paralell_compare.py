import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from gould_2026.estimator import Pipeline, CenteringEstimator, ArrayWithTime, KernelSmoother
from gould_2026.dimension_reduction.prosvd import proSVD
from gould_2026.dimension_reduction.jpca import sjPCA
from gould_2026.dimension_reduction.ica import mmICA
from gould_2026.prediction.bubblewrap import Bubblewrap
from gould_2026.regression import MultiKernelRegressor
from gould_2026.datasets import Zong22Dataset

# from gould_2026.utils import save_to_cache

def plot_history_with_tail(ax, data, current_t, tail_length=1, scatter_all=True, dim_1=0, dim_2=1, hist_bins=None, invisible=False, scatter_alpha=.1, scatter_s=5):
    """
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> X = np.random.normal(size=(100,2))
    >>> X = ArrayWithTime.from_notime(X)
    >>> plot_history_with_tail(ax, data=X, current_t=75, tail_length=4, scatter_alpha=1)
    """
    ax.cla()

    s = np.ones_like(data.t).astype(bool)
    if scatter_all:
        s = data.t <= current_t
    if hist_bins is None:
        ax.scatter(data[s,dim_1], data[s,dim_2], s=scatter_s, c='gray', edgecolors='none', alpha= 0 if invisible else scatter_alpha)
        back_color = 'white'
        forward_color = 'C0'
    else:
        s = s & np.isfinite(data).all(axis=1)
        ax.hist2d(data[s,dim_1], data[s,dim_2], bins=hist_bins)
        back_color = 'black'
        forward_color = 'white'


    linewidth = 2
    size = 10
    s = (current_t - tail_length < data.t) & (data.t <= current_t)
    ax.plot(data[s, dim_1], data[s, dim_2], color=back_color, linewidth=linewidth * 1.5, alpha= 0 if invisible else 1)
    ax.scatter(data[s, dim_1][-1], data[s, dim_2][-1], s=size * 1.5, color=back_color, alpha= 0 if invisible else 1)
    ax.plot(data[s, dim_1], data[s, dim_2], color=forward_color, linewidth=linewidth, alpha= 0 if invisible else 1)
    ax.scatter(data[s,dim_1][-1], data[s,dim_2][-1], color=forward_color, s=size, zorder=3, alpha= 0 if invisible else 1)
    ax.axis('off')



def plot_flow_fields(dim_reduced_data, x_direction=0, y_direction=1, grid_n=13, scatter_alpha=0, normalize_method=None, fig=None, axs=None, method='quiver', format_axis=True, limits=None, f_on_arrows=None):
    """
    Examples
    --------
    >>> X = np.random.normal(size=(100,2))
    >>> plot_flow_fields({'random points': X}, normalize_method='squares', grid_n=20)
    """
    assert normalize_method in {None, 'none', 'diffs', 'hcubes', 'squares'}
    if fig is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(dim_reduced_data), squeeze=False, layout='tight', figsize=(12,4))
        axs = axs[0]

    for idx, (name, latents) in enumerate(dim_reduced_data.items()):
        e1, e2 = np.zeros(latents.shape[1]), np.zeros(latents.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1

        ax: plt.Axes = axs[idx]
        ax.scatter(latents @ e1, latents @ e2, s=5, alpha=scatter_alpha)
        if limits is None:
            x1, x2, y1, y2 = ax.axis()
        else:
            x1, x2, y1, y2 = limits
        x_points = np.linspace(x1, x2, grid_n)
        y_points = np.linspace(y1, y2, grid_n)
        assert x1 < x2 and y1 < y2

        d_latents = np.diff(latents, axis=0)
        if normalize_method == 'diffs':
            d_latents = d_latents / np.linalg.norm(d_latents, axis=1)[:, np.newaxis]


        origins = []
        arrows = []
        n_points = []
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                proj_1 = (latents[:-1] @ e1)
                proj_2 = (latents[:-1] @ e2)
                # s stands for slice
                s = (
                        (x_points[i] <= proj_1) & (proj_1 < x_points[i + 1])
                        &
                        (y_points[j] <= proj_2) & (proj_2 < y_points[j + 1])
                )
                if s.sum():
                    arrow = np.nanmean(d_latents[s],axis=0)
                    if normalize_method == 'hcubes':
                        arrow = arrow / np.linalg.norm(arrow)
                    arrow = arrow
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())
                else:
                    arrow = np.nanmean(d_latents[s],axis=0) * 0
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())

        origins, arrows, n_points = np.array(origins), np.array(arrows), np.array(n_points)
        arrows = np.array([arrows @ e1, arrows @ e2]).T
        if normalize_method == 'squares':
            arrows = arrows / np.linalg.norm(arrows, axis=1)[:, np.newaxis]

        if f_on_arrows is not None:
            arrows = f_on_arrows(arrows)

        if method == 'quiver':
            ax.quiver(origins[:, 0], origins[:, 1], arrows[:,0], arrows[:,1], scale=1 / 20, units='dots', color='red')
        elif method == 'streamplot':
            origins = origins.reshape((grid_n-1,grid_n-1,2))
            a = origins[..., 1].mean(axis=0) # -1 2 is the x axis
            b = origins[..., 0].mean(axis=1)
            arrows = arrows.reshape((grid_n-1,grid_n-1,2))
            ax.streamplot(y=a, x=b, v=arrows[...,1].T, u=arrows[...,0].T, color='red')
        else:
            raise ValueError()

        if format_axis:
            ax.axis('scaled')
            ax.axis('off')



def main():
    d = Zong22Dataset()


    prosvd_k = 8

    # @save_to_cache("parallel_compare")
    def f():
        p = Pipeline([CenteringEstimator(), KernelSmoother(tau=2 * .68 / d.neural_data.dt), proSVD(k=prosvd_k)])

        dim_red_methods = [Pipeline(), sjPCA(), mmICA()]
        # predictors = [StreamingKalmanFilter(log_level=2, check_dt=True, n_steps_to_predict=1, steps_between_refits=50) for _ in dim_red_methods]
        predictors = [Bubblewrap(log_level=2, check_dt=True, n_steps_to_predict=1) for _ in dim_red_methods]
        # predictors = [VJF(log_level=2, check_dt=True, n_steps_to_predict=1) for _ in dim_red_methods]

        regs = [MultiKernelRegressor(maxlen=10000, length_scales=[0.1725], reweight_every=np.inf) for _ in dim_red_methods]

        outputs = [[] for _ in dim_red_methods]

        pbar = tqdm(total=round(d.neural_data.t.max(),2))
        for data in p.streaming_run_on(d.neural_data):

            metrics = []
            in_space_data = []
            for dim_red_method, predictor, output_accumulator in zip(dim_red_methods, predictors, outputs):
                in_space_datum = dim_red_method.step(data)
                in_space_datum = in_space_datum[:,:4]
                in_space_data.append(in_space_datum)
                output_accumulator.append(in_space_datum)

                mse = ((in_space_datum - predictor.predict(1)) ** 2).mean()
                neg_log_pred_p = -predictor.unevaluated_log_pred_p(1)(in_space_datum)
                metrics.append(neg_log_pred_p)
                predictor.step(in_space_datum)

            best_regressor = np.argmin(metrics)
            for i, (reg, in_space_datum) in enumerate(zip(regs, in_space_data)):
                reg.observe(in_space_datum, np.array([i == best_regressor]))

            pbar.update(round(data.t,2) - pbar.n)


        outputs = [ArrayWithTime.from_list(o, drop_early_nans=True, squeeze_type='to_2d') for o in outputs]
        return outputs, dim_red_methods, regs

    outputs, dim_red_methods, regs = f()



    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter

    labels = ['prosvd','sjpca','mmica']

    def make_heatmap(ax, o, x_direction, y_direction, color_direction, density=13, limits=None, sigma=1, cax=None):
        e1, e2, ec = np.zeros(o.shape[1]), np.zeros(o.shape[1]), np.zeros(o.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1
        ec[color_direction] = 1
        x = o @ e1
        y = o @ e2
        c = o @ ec

        # ax.scatter(x, y, c=c, s=1, cmap='plasma')

        if limits is None:
            axis = ax.axis()
        else:
            axis = limits

        x_edges = np.linspace(axis[0], axis[1], density + 1)
        y_edges = np.linspace(axis[2], axis[3], density + 1)
        x_centers = np.convolve(x_edges, [0.5, 0.5], mode='valid')
        y_centers = np.convolve(y_edges, [0.5, 0.5], mode='valid')

        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        c_grid = np.zeros_like(x_grid)

        for i in range(len(y_centers)):
            for j in range(len(x_centers)):
                slice_1 = (x_edges[j] < x) & (x < x_edges[j+1])
                slice_2 = (y_edges[i] < y) & (y < y_edges[i+1])
                s = (slice_1 & slice_2)
                if s.sum()<1:
                    c_grid[i,j] = 0
                else:
                    c_grid[i,j] = np.mean(c[s])

        c_grid = gaussian_filter(c_grid,sigma=sigma, mode='constant', cval=0.0)

        cmap = plt.colormaps['plasma']
        cmap.set_bad('k')
        cmesh = ax.pcolormesh(x_grid, y_grid, c_grid, cmap=cmap, vmin=0, vmax=.58)
        print(f'{c_grid.min()=:.2f} {c_grid.max()=:.2f}')
        if cax is not None:
            fig.colorbar(cmesh, cax=cax, orientation='horizontal')



    fig, axs = plt.subplots(3, len(dim_red_methods), figsize=(10, 5), squeeze=False, sharex='col', sharey='col', layout='constrained', height_ratios=[.1, 1,1])
    gs = axs[0,0].get_gridspec()

    for ax in axs[0,:]:
        ax.remove()
    cax = fig.add_subplot(gs[0,:])

    x_direction = 0
    y_direction = 1

    mmica_limits = [-5, 5.1, -6.7, 8]
    plot_limits = [None, None, mmica_limits]
    for k,v,limit,ax,arrow_scale in zip(labels, outputs, plot_limits, axs[1], [.75,1,1]):
        plot_flow_fields(
            {k:v},
            # method='streamplot',
            method='quiver',
            grid_n=20,
            # normalize_method='diffs',
            # normalize_method='none',
            normalize_method='diffs',
            # normalize_method='hcubes',
            fig=fig, axs=[ax], format_axis=False,
            x_direction=x_direction, y_direction=y_direction, scatter_alpha=0,
            limits=limit,
            f_on_arrows=lambda x: x * arrow_scale *.8,
        )



    for reg, ax, l, o in zip(regs, axs[1], plot_limits, outputs):
        # ax.scatter(reg.history[:,x_direction], reg.history[:,y_direction], c=reg.history[:,-1], s=1, cmap='plasma', vmin=-.1, vmax=1.1)

        stride = 10
        base = 17 * stride
        to_plot = o.slice_by_time(slice(base, base+stride+2.5))
        line = ax.plot(to_plot[:,x_direction], to_plot[:,y_direction], 'k', lw=1.25)

        for arrow_index in [45]:
            ax.annotate('',
                             xytext=(to_plot[arrow_index, x_direction], to_plot[arrow_index, y_direction]),
                             xy=(to_plot[arrow_index+1, x_direction], to_plot[arrow_index+1, y_direction]),
                             arrowprops=dict(arrowstyle="simple", color='k'),
                             size=11
                             )

        ax.axis(l)



    for reg, ax, old_ax in zip(regs, axs[2], axs[1]):
        density = 200
        make_heatmap(ax, reg.input_histories[0], x_direction, y_direction, color_direction=-1, density=density, sigma=density * 4/200, cax=cax)

    return fig

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig = main()

    fig.savefig(args.output, bbox_inches="tight")

