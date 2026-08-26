from enum import Enum
import matplotlib
import matplotlib.pyplot as plt

DPI = 600
LINEWIDTH = 0.5
AXES_LINEWIDTH = 0.25
EM = 6.

matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['savefig.dpi'] = DPI
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0


matplotlib.rcParams['figure.dpi'] = DPI
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 1.25*EM
matplotlib.rcParams['axes.labelpad'] = .4*EM
matplotlib.rcParams['axes.labelsize'] = 1.25*EM


matplotlib.rcParams['lines.linewidth'] = LINEWIDTH
matplotlib.rcParams['lines.markersize'] = LINEWIDTH*4

matplotlib.rcParams['axes.linewidth'] = AXES_LINEWIDTH


matplotlib.rcParams['patch.linewidth'] = AXES_LINEWIDTH
# matplotlib.rcParams['legend.frameon'] = False

for axis in ['x', 'y']:

    matplotlib.rcParams[f'{axis}tick.labelsize'] = EM
    matplotlib.rcParams[f'{axis}tick.major.width'] = AXES_LINEWIDTH
    matplotlib.rcParams[f'{axis}tick.major.size'] = EM/3
    matplotlib.rcParams[f'{axis}tick.major.pad'] = EM/3
    matplotlib.rcParams[f'{axis}tick.direction'] = 'in'


# matplotlib.rcParams['constrained_layout.use'] = True

overused_red = '#ca1469ff'

class Palette(str, Enum):
    overused_red = overused_red
    gray_background = '#E6E6E6FF'

    # dimension reduction algorithms
    prosvd = "#EC3C8E"
    sjpca = "#F94B00"
    mmica = "#2FA194"

    # 1-step-ahead prediction plots
    stim_regressed = overused_red
    blind = '#4d4d4dff'

    # open vs closed loop
    open_trivial = '#000000'
    open_nontrivial = '#069406ff'
    # closed_trivial = '#000000'
    closed_nontrivial = overused_red

    # sim-stim targets
    Q_0 = overused_red
    feasible = '#beaed4ff'
    random = '#e1ab77ff'
    dense = '#6794cfff'
    negative = '#7ec97eff'


    f_hat = '#000000'  # gray
    s_designed = overused_red
    s_obs = overused_red
    v = '#1e7608ff'
    x = '#000000'
    u = '#000000'


    kalman = '#000000'
    bubblewrap = '#000000'
    vjf = '#000000'
