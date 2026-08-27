from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
from contextlib import contextmanager

LINEWIDTH = 0.5
AXES_LINEWIDTH = 0.25
EM = 6.

matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0

@contextmanager
def paper_plot_context(frameon=True):
    rcParams = dict()

    rcParams['figure.dpi'] = 83 # this renders true-to-scale with the QT backend in pycharm + jupyter on Tycho
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
    rcParams['legend.frameon'] = frameon

    for axis in ['x', 'y']:
        rcParams[f'{axis}tick.labelsize'] = EM
        rcParams[f'{axis}tick.major.width'] = AXES_LINEWIDTH
        rcParams[f'{axis}tick.major.size'] = EM / 3
        rcParams[f'{axis}tick.major.pad'] = EM / 3
        rcParams[f'{axis}tick.direction'] = 'in'


    with plt.rc_context(rc=rcParams):
        yield


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
    open_trivial = black
    open_nontrivial = '#069406ff'
    # closed_trivial = '#000000'
    closed_nontrivial = overused_red

    # sim-stim targets
    Q_0 = overused_red
    feasible = '#beaed4ff'
    random = '#e1ab77ff'
    dense = '#6794cfff'
    negative = '#7ec97eff'


    f_hat = black
    s_designed = overused_red
    s_obs = overused_red
    v = '#1e7608ff'
    x = black
    u = black


    kalman = black
    bubblewrap = black
    vjf = black
