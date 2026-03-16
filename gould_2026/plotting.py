from enum import Enum
import matplotlib
matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['savefig.dpi'] = 300

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
    Q0 = overused_red
    feasible = '#beaed4ff'
    random = '#e1ab77ff'
    dense = '#6794cfff'
    neg = '#7ec97eff'


    f_hat = '#000000'  # gray
    s_hat = overused_red
    s_obs = overused_red
    v = '#1e7608ff'
    x = '#000000'
    u = '#000000'


    kalman = '#000000'
    bubblewrap = '#000000'
    vjf = '#000000'
