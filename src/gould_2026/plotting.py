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

class Palette(str, Enum):
    gray_background = '#E6E6E6FF'

    # dimension reduction algorithms
    prosvd = "#EC3C8E"
    sjpca = "#F94B00"
    mmica = "#2FA194"

    # 1-step-ahead prediction plots
    stim_regressed = overused_red
    blind = '#4d4d4dff'

    # open vs closed loop
    open_trivial = '#fc7f00ff'
    open_nontrivial = '#ffce0aff'
    # closed_trivial = '#000000'
    closed_nontrivial = overused_red
    random_trivial = blind

    # sim-stim targets
    Q_0 = '#c87a65'
    feasible = '#beaed4ff'
    random = '#e1ab77ff'
    dense = '#6794cfff'
    negative = '#7ec97eff'


    f_hat = "#525252"
    s_designed = 'red'
    s_obs = overused_red
    v = '#1e7608ff'
    x = black
    u = black


    kalman = black
    bubblewrap = black
    vjf = black

    @staticmethod
    def draelos_25_cmap():
        sub_min = 0.0
        sub_max = 114 / 202
        orig_cmap = plt.colormaps["cool_r"]
        extracted_colors = orig_cmap(np.linspace(sub_min, sub_max, 256))
        sub_cmap = mcolors.LinearSegmentedColormap.from_list("cool_subset", extracted_colors)
        new_norm = mcolors.Normalize(vmin=-32, vmax=82)
        cmap_f = lambda x: sub_cmap(new_norm(x))

        return sub_cmap, new_norm, cmap_f

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
    sns.stripplot(data, x=x, y=y, hue=x, palette=palette, s=LINEWIDTH*5)
    ax.set_xticklabels(['', ''])
    ax.set_yticks(yticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig

def make_scatter_violin(df, keys, pairs=None, palette=None, xticks=None):
    if palette is None:
        palette = {
            'open id': Palette.open_trivial,
            'open flip': Palette.open_nontrivial,
            'closed flip': Palette.closed_nontrivial,
            'rand id': Palette.random_trivial,
        }
    fig, axs = plt.subplots(figsize=np.array([3, 2])*5/6, nrows=1, ncols=2, width_ratios=[2,1], sharey=True, squeeze=False, layout='constrained')

    ax = axs[0,0]
    for k in keys:
        sub_df = df[df.sr_key == k]
        sub_df = sub_df[sub_df.t > sub_df.t.median()]

        ax.scatter(sub_df['r'],sub_df['theta'], LINEWIDTH*2, color=palette[k])

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
