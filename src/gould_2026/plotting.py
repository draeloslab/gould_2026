from enum import Enum
import matplotlib
from matplotlib.path import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

LINEWIDTH = 0.5
AXES_LINEWIDTH = 0.25
EM = 6.

rcParams = dict()

rcParams['figure.dpi'] = 200
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

rcParams['font.sans-serif'] = 'Arial'
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 1.25 * EM
rcParams['axes.labelpad'] = .4 * EM
rcParams['axes.labelsize'] = 1.25 * EM

rcParams['lines.linewidth'] = LINEWIDTH
rcParams['lines.markersize'] = LINEWIDTH * 4

rcParams['axes.linewidth'] = AXES_LINEWIDTH

rcParams['patch.linewidth'] = AXES_LINEWIDTH

for axis in ['x', 'y']:
    rcParams[f'{axis}tick.labelsize'] = EM
    rcParams[f'{axis}tick.major.width'] = AXES_LINEWIDTH
    rcParams[f'{axis}tick.major.size'] = EM / 3
    rcParams[f'{axis}tick.major.pad'] = EM / 3
    rcParams[f'{axis}tick.direction'] = 'out'


matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0


@contextmanager
def paper_plot_context(overrides=None):
    if overrides is None:
        overrides = dict()

    # rcParams['legend.frameon'] = frameon
    with plt.rc_context(rc=rcParams | overrides):
        yield

def use_paper_plot_context():
    plt.rcParams.update(rcParams)


overused_red = '#ca1469ff'

black = '#000000'
neutral_gray = '#ababab'
transparent = '#00000000'

stim_color = '#c6244aff'
response_color = '#26956aff'

class Palette(str, Enum):
    gray_background = '#E6E6E6FF'
    stim_color = '#c6244aff'

    # 1-step-ahead prediction plots
    stim_regressed = response_color
    blind = neutral_gray

    # open vs closed loop
    open_trivial = stim_color
    open_nontrivial = stim_color
    closed_nontrivial = stim_color
    random_trivial = stim_color

    # sim-stim targets
    Q_0 = stim_color
    feasible = stim_color
    random = stim_color
    dense = stim_color
    negative = stim_color


    f_hat = "#525252"
    s_designed = stim_color
    s_obs = stim_color
    v = '#1e7608ff'
    x = black
    u = black


    @classmethod
    def draelos_25_cmap(cls):
        return cls.viridis_cmap()


    @staticmethod
    def cool_cmap():
        sub_min = 0.0
        sub_max = 114 / 202
        orig_cmap = plt.colormaps["cool_r"]
        extracted_colors = orig_cmap(np.linspace(sub_min, sub_max, 256))
        sub_cmap = mcolors.LinearSegmentedColormap.from_list("cool_subset", extracted_colors)
        new_norm = mcolors.Normalize(vmin=-32, vmax=82)
        cmap_f = lambda x: sub_cmap(new_norm(x))
        return sub_cmap, new_norm, cmap_f

    @staticmethod
    def viridis_cmap():
        r = np.array([0.0, 114 / 202]) +.2
        orig_cmap = plt.colormaps["viridis"]
        extracted_colors = orig_cmap(np.linspace(r[0], r[1], 256))
        sub_cmap = mcolors.LinearSegmentedColormap.from_list("viridis_subset", extracted_colors)
        new_norm = mcolors.Normalize(vmin=-32, vmax=82)
        cmap_f = lambda x: sub_cmap(new_norm(x))
        return sub_cmap, new_norm, cmap_f


class Markers(str, Enum):
    open_trivial = 'X'
    open_nontrivial = 'd'
    closed_nontrivial = 'o'
    random_trivial = '.'

def make_violinplot_inner_kws():
    _vh = .5
    verts = [ (-1., -_vh), (-1., _vh), (1., _vh), (1., -_vh), (-1., -_vh), ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
    white_bar_path = Path(verts, codes)
    violinplot_inner_kws = {'marker': white_bar_path, 'markersize': LINEWIDTH*2, 'markerfacecolor': 'white', }
    return violinplot_inner_kws



def make_boxplot(data, palette, yticks, x=None, y=None):
    data = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(1.7,1.7))
    sns.boxplot(data, x=x, y=y, hue=x, palette=palette, boxprops={'alpha':.5}, fliersize=0, linewidth=LINEWIDTH)
    np.random.seed(0)
    sns.stripplot(data, x=x, y=y, hue=x, palette=palette, s=LINEWIDTH*5)
    ax.set_xticklabels(['', ''])
    ax.set_yticks(yticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig

def make_scatter_violin(df, keys, pairs=None, palette=None, markers=None, xticks=None):
    if palette is None:
        palette = {
            'open id': Palette.open_trivial,
            'open flip': Palette.open_nontrivial,
            'closed flip': Palette.closed_nontrivial,
            'rand id': Palette.random_trivial,
        }

    if markers is None:
            markers = {
            'open id': (Markers.open_trivial, stim_color, stim_color),
            'open flip': (Markers.open_nontrivial, stim_color, stim_color),
            'closed flip': (Markers.closed_nontrivial, stim_color, stim_color),
            'rand id': (Markers.random_trivial, stim_color, stim_color),
        }
    fig, axs = plt.subplots(figsize=np.array([3, 2])*5/6, nrows=1, ncols=2, width_ratios=[2,1], sharey=True, squeeze=False, layout='constrained')

    ax = axs[0,0]
    for k in keys:
        sub_df = df[df.sr_key == k]
        sub_df = sub_df[sub_df.t > sub_df.t.median()]

        ax.scatter(sub_df['r'],sub_df['theta'], s=LINEWIDTH*6, color=palette[k], marker=markers[k][0], edgecolors=markers[k][1], facecolors=markers[k][2], alpha=1)

        ax.set_ylim(0, 95)
        ax.axhline(90, linestyle='--', color='gray', alpha=0.5)

    ax.set_ylabel(r'$\theta_{\text{err}}$')

    ax.set_xlabel(r'$\left\| s_{\text{obs}} \right\|$')
    ax.set_yticks([0, 45,  90, 135, 180])
    if xticks:
        ax.set_xticks(xticks)


    ax = axs[0,1]
    sub_df = df[df['sr_key'].apply(lambda x: x in keys)]
    sns.violinplot(data=sub_df, x='sr_key', y='theta', hue='sr_key', palette=palette, cut=0, inner_kws=make_violinplot_inner_kws(), ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels(['', ''])

    if pairs:
        annotator = Annotator(ax, data=sub_df, pairs=pairs, x='sr_key', y='theta')
        annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
        annotator.apply_and_annotate()
    return fig

def make_latent_quiver(latent_1, latent_2, pre_points, latent_deltas, cmap_f, delta_scale=1, too_short_cutoff=0):
    fig, ax = plt.subplots(figsize=np.array([1.5, 1.5]), constrained_layout=True)
    ax.scatter(latent_1, latent_2, c='gray', alpha=0.4, s=LINEWIDTH* 2/1.5, rasterized=True)

    for pre_point, delta in zip(pre_points, latent_deltas):
        angle = np.atan2(delta[1], delta[0]) * 180/np.pi
        color=cmap_f(angle)
        delta = delta * delta_scale

        if np.linalg.norm(delta[:2]) > too_short_cutoff:
            ax.quiver(pre_point[0], pre_point[1], delta[0], delta[1], angles='xy', scale_units='xy', scale=1.5, width=.007, color=color)
        else:
            ax.scatter(pre_point[0], pre_point[1], s=5/1.5 * LINEWIDTH, color=color)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return fig, ax

def make_pred_heatmap_after_quiver(latent_0, latent_1, quiver_axis, f,cmap_f, center, e1, e2):
    depth = 100
    x_edges = np.linspace(quiver_axis[0], quiver_axis[1], depth + 1)
    y_edges = np.linspace(quiver_axis[2], quiver_axis[3], depth + 1)
    x = np.convolve(x_edges, [.5, .5], 'valid')
    y = np.convolve(y_edges, [.5, .5], 'valid')

    k = 10

    points = np.array(np.meshgrid(x, y)).T @ np.vstack([e1, e2]) + center
    points = points.transpose((1, 0, 2))
    predictions = f(points.reshape((-1, k)))
    average_predictions = np.mean(predictions, axis=1)
    angles = np.atan2(average_predictions @ e2, average_predictions @ e1) * 180 / np.pi
    lengths = np.linalg.norm(average_predictions[:, :2], axis=1)
    color = np.array([cmap_f(a) for a in angles])
    c = color.reshape((depth, depth, -1))

    fig, ax = plt.subplots(figsize=np.array([1.5, 1.5]), constrained_layout=True)
    m = ax.pcolormesh(x_edges, y_edges, c, rasterized=True)
    ax.scatter(latent_0, latent_1, c='k', alpha=0.2, s=LINEWIDTH * 1 / 1.5)
    ax.set_xlim(x_edges.min(), x_edges.max())
    ax.axis(quiver_axis)
    ax.axis('off')
    return fig, ax
