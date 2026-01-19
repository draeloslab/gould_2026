import warnings
from PIL import Image
import pims
from pynwb import NWBHDF5IO
import h5py
import fsspec
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from contextlib import contextmanager
from abc import ABC, abstractmethod
import pandas as pd

import numpy as np
from scipy.stats import special_ortho_group

from .estimator import ArrayWithTime

from .prediction.kalman_filter import KalmanFilter

import pathlib

DATA_BASE_PATH = pathlib.Path(__file__).parent.parent.absolute() / "data"


class LDS:
    def __init__(self, A, C, W, Q, B=None, state_center=None, observation_center=None):
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q

        if (self.W == 0).all():
            self.W_cholesky = 0 * self.W
        else:
            self.W_cholesky = np.linalg.cholesky(self.W)

        if (self.Q == 0).all():
            self.Q_cholesky = 0 * self.Q
        else:
            self.Q_cholesky = np.linalg.cholesky(self.Q)

        self.B = B if B is not None else np.zeros((0, A.shape[0]))
        self.state_center = state_center if state_center is not None else 0
        self.observation_center = observation_center if observation_center is not None else 0
        self.check_shapes_correct()

    def check_shapes_correct(self):
        assert self.A is not None
        assert self.A.shape == self.W.shape
        assert self.A.shape[1] == self.C.shape[0] == self.B.shape[1]
        assert self.C.shape[1] == self.Q.shape[1] == self.Q.shape[0]
        assert np.allclose(self.Q, self.Q.T)
        assert np.allclose(self.W, self.W.T)

    def simulate(self, n_steps, initial_state=None, U=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if isinstance(U, np.ndarray):
            assert U.shape[0] == n_steps
            def u_function(lds, state, i, rng):
                return U[i]
        else:
            u_function = U

        states = np.zeros((n_steps, self.A.shape[0]))
        observations = np.zeros((n_steps, self.C.shape[1]))
        control = np.zeros((n_steps, self.B.shape[0]))

        if initial_state is not None:
            states[0] = np.array(initial_state) - self.state_center
        else:
            states[0,:] = 0
            warnings.warn("simulating with the initial state in equilibrium")

        state = states[0]
        for i in range(n_steps):
            state, observation, u = self.simulate_step(state, rng, u_function, i, use_state_dynamics=i != 0, add_centers=False)
            states[i] = state
            observations[i] = observation
            control[i] = u

        return states + self.state_center, observations + self.observation_center, control
        # return ArrayWithTime.from_notime(states + self.state_center), ArrayWithTime.from_notime(observations + self.observation_center), ArrayWithTime.from_notime(control)

    def simulate_step(self, state, rng, u_function=None, i=None, use_state_dynamics=True, add_centers=True):
        if add_centers:
            state = state - self.state_center

        u = np.array([])
        if u_function is not None and isinstance(u_function, np.ndarray):
            u = u_function
        elif u_function is not None:
            u = u_function(lds=self, state=state, i=i, rng=rng)

        if use_state_dynamics:  # I don't want this sometimes on the first iteration
            state = state @ self.A
            random_jitter = rng.normal(size=self.A.shape[1]) @ self.W_cholesky
            state = state + random_jitter

        state += u @ self.B

        observation = state @ self.C
        observation_noise = rng.normal(size=self.C.shape[1]) @ self.Q_cholesky
        observation = observation + observation_noise

        if add_centers:
            state = state + self.state_center
            observation = observation + self.observation_center

        return state, observation, u


    @classmethod
    def from_kalman_filter(cls, kf):
        kf: KalmanFilter
        return cls(kf.A, kf.C, kf.W, kf.Q, state_center=kf.X_mean, observation_center=kf.Y_mean)

    @classmethod
    def circular_lds(cls, transitions_per_rotation=30 + 1 / np.pi, obs_d=10, process_noise=0.01, obs_noise=0.02, obs_center=0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        theta = 2*np.pi/transitions_per_rotation
        A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        C = special_ortho_group(dim=obs_d, seed=rng).rvs()[:, :2].T
        lds = cls(A, C, np.eye(2) * process_noise, np.eye(obs_d) * obs_noise, state_center=0, observation_center=obs_center)
        lds.transitions_per_rotation = transitions_per_rotation
        return lds

    @classmethod
    def nest_lds(cls, transitions_per_rotation=30 + 1 / np.pi, rng=None, noise=0.05):
        rng = rng if rng is not None else np.random.default_rng()
        base_lds = LDS.circular_lds(transitions_per_rotation=transitions_per_rotation, rng=rng)

        A = base_lds.A
        A = np.hstack([A, np.zeros((A.shape[0], 1))])
        A = np.vstack([A, np.zeros((1, A.shape[1]))])
        A[-1, -1] = .8
        C = np.eye(A.shape[1])
        B = np.eye(A.shape[1])
        W = np.eye(A.shape[1]) * noise
        Q = np.eye(A.shape[1]) * noise
        return LDS(A, C, W, Q, B=B)

    @classmethod
    def run_nest_dynamical_system(cls, rotations, transitions_per_rotation=30 + 1 / np.pi, stim_magnitude=1, stims_per_rotation=1, radius=5, u_function=None, rng=None, early_shift=1e-12, noise=0.05, theta_0=None):
        rng = rng if rng is not None else np.random.default_rng()
        dynamics_rng, stim_rng = rng.spawn(2)
        if theta_0 is None:
            theta_0 = dynamics_rng.uniform(0, 2 * np.pi)
        lds = cls.nest_lds(transitions_per_rotation=transitions_per_rotation, rng=dynamics_rng, noise=noise)
        N = int(rotations * transitions_per_rotation)
        t = np.linspace(0, N / transitions_per_rotation, N)

        stim = t * 0
        stim[stim_rng.choice(stim.shape[0], size=int(stims_per_rotation * N / transitions_per_rotation), replace=False)] = 1

        if u_function == 'curvy':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2])
                return u
        elif u_function == 'curvy flips':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2]) * (-1 if i > stim.shape[0]//2 else 1)
                return u
        elif u_function == 'curvy spins':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])

                state = np.array(state)

                transition1 = 25 * transitions_per_rotation
                transition2  = 45 * transitions_per_rotation
                if i <= transition1:
                    rotation_angle = 0
                elif transition1 < i <= transition2:
                    rotation_angle = np.pi
                elif transition2 < i:
                    rotation_angle = (i-transition2) * 2*np.pi / (30 * transitions_per_rotation) + np.pi
                else:
                    raise ValueError()

                rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                            [np.sin(rotation_angle),  np.cos(rotation_angle)]])
                state[:2] = rotation_matrix @ state[:2]

                u[2] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2])
                return u
        elif u_function == 'curvy spins alld-resp':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])

                state = np.array(state)

                transition1 = 25 * transitions_per_rotation
                transition2  = 45 * transitions_per_rotation
                if i <= transition1:
                    rotation_angle = 0
                elif transition1 < i <= transition2:
                    rotation_angle = np.pi
                elif transition2 < i:
                    rotation_angle = (i-transition2) * 2*np.pi / (30 * transitions_per_rotation) + np.pi
                else:
                    raise ValueError()

                rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                            [np.sin(rotation_angle),  np.cos(rotation_angle)]])
                state[:2] = rotation_matrix @ state[:2]

                u[:] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2])
                return u

        elif u_function == 'constant':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i]
                return u
        elif u_function is None:
            u_function = lambda **_: np.zeros(lds.B.shape[0])

        states, observations, received_stim = lds.simulate(N, initial_state=[radius * np.cos(theta_0), radius * np.sin(theta_0), 0], U=u_function, rng=dynamics_rng)

        assert early_shift == 0 or np.diff(t).mean() / early_shift > 100

        stim = ArrayWithTime(stim[:,None], t - 2*early_shift)
        X = ArrayWithTime(states, t - 1*early_shift)
        Y = ArrayWithTime(observations, t - 0*early_shift)

        return X, Y, stim


def generate_circle_embedded_in_high_d(rng, m=1000, n=4, stddev=1, transitions_per_rotation=10):
    lds = LDS.circular_lds(transitions_per_rotation=transitions_per_rotation, obs_d=n, process_noise=0, obs_noise=stddev, rng=rng)
    _, X_all, _ = lds.simulate(m, initial_state=np.array([10,0]), rng=rng)
    X_dot = np.diff(X_all, axis=0)
    X = X_all[1:]
    return X, X_dot, dict(C=lds.C.T)



class DandiDataset:
    # TODO: should name be DANDIDataset?
    automatically_downloadable = True

    @property
    @abstractmethod
    def dandiset_id(self):
        pass

    @property
    @abstractmethod
    def version_id(self):
        pass

    @contextmanager
    def acquire(self, asset_path):
        # https://pynwb.readthedocs.io/en/latest/tutorials/advanced_io/streaming.html
        with DandiAPIClient() as client:
            asset = client.get_dandiset(self.dandiset_id, version_id=self.version_id).get_asset_by_path(asset_path)
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        fs = fsspec.filesystem("http")
        fs = CachingFileSystem(
            fs=fs,
            cache_storage=[DATA_BASE_PATH / "nwb_cache"],
        )

        with fs.open(s3_url, "rb") as f:
            with h5py.File(f) as file:
                fhan = NWBHDF5IO(file=file, load_namespaces=True)
                yield fhan


class Odoherty21Dataset(DandiDataset):
    doi = 'https://dandiarchive.org/dandiset/000129/draft/'
    dandiset_id = "000129"
    version_id = None

    dataset_base_path = DATA_BASE_PATH / "odoherty21"
    automatically_downloadable = True

    def __init__(self, bin_width=0.03, downsample_behavior=True, neural_lag=0, drop_third_coord=True, pos_rescale_factor=1, vel_rescale_factor=1):
        self.bin_width = bin_width
        self.downsample_behavior = downsample_behavior
        self.drop_third_coord = drop_third_coord
        self.neural_lag = neural_lag
        self.pos_rescale_factor = pos_rescale_factor
        self.vel_rescale_factor = vel_rescale_factor
        assert self.neural_lag >= 0

        self.units, self.finger_pos, self.finger_vel, self.finger_t, A, bin_ends = self.construct()
        self.neural_data = ArrayWithTime(A, bin_ends)
        self.behavioral_data = ArrayWithTime(self.finger_pos, self.finger_t)

        self.beh_pos = ArrayWithTime(self.finger_pos, self.finger_t)
        self.beh_vel = ArrayWithTime(self.finger_vel, self.finger_t)
        self.beh_pos_vel = ArrayWithTime(np.hstack([self.finger_pos, self.finger_vel]), self.finger_t)

    def construct(self):
        # TODO: get the warnings to work again
        # with warnings.catch_warnings(record=True) as warning_list:
        with self.acquire("sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb") as fhan:
            ds = fhan.read()
            units = ds.units.to_dataframe()
            finger_pos = ds.processing['behavior'].data_interfaces['finger_pos'].data[:]
            finger_pos_t = np.arange(finger_pos.shape[0]) * ds.processing['behavior'].data_interfaces['finger_pos'].conversion
            finger_vel = ds.processing['behavior'].data_interfaces['finger_vel'].data[:]
            finger_vel_t = np.arange(finger_vel.shape[0]) * ds.processing['behavior'].data_interfaces['finger_vel'].conversion

            # for w in warning_list:
            #     if self.supress_warnings and "Ignoring cached namespace" in str(w):
            #         continue
            #     warnings.warn_explicit(message=w.message, category=w.category, filename=w.filename, lineno=w.lineno, source=w.source)

        start_time = units.iloc[0, 2].min()
        end_time = units.iloc[0, 2].max()
        bins = np.arange(start_time, end_time, self.bin_width)
        bin_ends = bins[1:]

        A = np.zeros(shape=(bins.shape[0] - 1, len(units)))

        for i, (_, row) in enumerate(units.iterrows()):
            A[:, i], _ = np.histogram(row['spike_times'], bins=bins)


        factor = 4
        if self.downsample_behavior:
            finger_pos = finger_pos[::factor]
            finger_pos_t = finger_pos_t[::factor]
            finger_vel = finger_vel[::factor]
            finger_vel_t = finger_vel_t[::factor]

        bin_ends = bin_ends + self.neural_lag
        assert (finger_pos_t == finger_vel_t).all()
        finger_t = finger_pos_t

        if self.drop_third_coord:
            finger_pos = finger_pos[:,:2]
            finger_vel = finger_vel[:,:2]

        finger_pos = finger_pos * self.pos_rescale_factor
        finger_vel = finger_vel * self.vel_rescale_factor


        return units, finger_pos, finger_vel, finger_t, A, bin_ends

    def plot_variances(self, ax):
        ax.hist(np.squeeze(self.neural_data.a).std(axis=0), bins=50, label='neural')
        for x in np.nanstd(np.squeeze(self.beh_pos.a), axis=0):
            ax.axvline(x, color='C1', label='pos')

        for x in np.nanstd(np.squeeze(self.beh_vel.a), axis=0):
            ax.axvline(x, color='C2', label='vel')

        ax.legend()
        ax.set_xlabel('variance')
        ax.set_ylabel('count')

class Zong22Dataset:
    doi = "https://dx.doi.org/10.11582/2022.00008"
    automatically_downloadable = False
    dataset_base_path = DATA_BASE_PATH / 'zong22'

    def make_cookie_entry(area, animal_id, date, f_part, f_total, cookie_status, filtered):
        assert type(area) == str  # this can become static after 3.10
        cookie_status = 'with' if cookie_status else 'no'
        filtered = 'filtered' if filtered else ''
        return {
            'basepath':       f'{area}_recordings/{animal_id}/{date}/',
            'raw_frames':     f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001.tif',
            'behavior_csv':   f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001_trackingVideoDLC_resnet50_OPENMINI2P_bottomcameraAug26shuffle1_1030000{filtered}.csv',
            'behavior_video': f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001_trackingVideo.avi',
            'part_of_F': (f_part,f_total)
        }

    def make_object_entry(area, animal_id, date, f_part, f_total, object_n, filtered):
        assert type(area) == str  # this can become static after 3.10
        object_str = f'object{object_n}' if object_n is not None else 'noobject'
        filtered = 'filtered' if filtered else ''
        return {
            'basepath':       f'{area}_recordings/{animal_id}/{date}/',
            'raw_frames':     f'{animal_id}_imaging_{date}_{object_str}_00001.tif',
            'behavior_csv':   f'{animal_id}_imaging_{date}_{object_str}_00001_trackingVideoDLC_resnet50_OPENMINI2P_bottomcameraAug26shuffle1_1030000{filtered}.csv',
            # 'behavior_video': f'{animal_id}_imaging_{date}_{object_str}_00001_trackingVideo.avi',
            'part_of_F': (f_part,f_total)
        }

    sub_datset_info = pd.DataFrame([
        make_cookie_entry('VC', '93562', '20200817', 1, 2, False, True),
        make_cookie_entry('VC', '93562', '20200817', 2, 2, True, True),

        make_cookie_entry('MEC', '94557', '20200822', 1, 2, False, False),
        make_cookie_entry('MEC', '94557', '20200822', 1, 2, True, False),

        make_object_entry('MEC', '94557', '20201008', 1, 3, None, True),
        make_object_entry('MEC', '94557', '20201008', 2, 3, 1, True),
        make_object_entry('MEC', '94557', '20201008', 3, 3, 2, True),
    ])

    sub_datasets = list(sub_datset_info.index)

    def __init__(self, sub_dataset_identifier=sub_datasets[0], neural_lag=0, neural_scale=1, pos_scale=1, hd_scale=1, h2b_scale=1):
        if isinstance(sub_dataset_identifier, int):
            sub_dataset_identifier = self.sub_datasets[sub_dataset_identifier]

        self.sub_dataset = sub_dataset_identifier
        self.neural_Fs = 15
        self.neural_lag = neural_lag
        self.neural_scale = neural_scale
        self.bin_width = 1/self.neural_Fs  # todo: make this universal?
        self.F, self.raw_images, self.behavior_video, self.behavior_df, self.n_cells, self.stat, self.ops = self.acquire()


        self.neural_data = ArrayWithTime(self.F.T * self.neural_scale, (np.arange(self.F.shape[1]) * 1 / self.neural_Fs) + self.neural_lag)
        self.behavioral_data = ArrayWithTime(self.behavior_df.loc[:, ['x', 'y', 'hd', 'h2b']] * np.array([pos_scale, pos_scale, hd_scale, h2b_scale]), self.behavior_df.loc[:, 't'])

        self.video_t = np.squeeze(self.behavioral_data.t)

    def acquire(self):
        sub_dataset_base_path = self.dataset_base_path / self.sub_datset_info.basepath[self.sub_dataset]
        if not sub_dataset_base_path.is_dir():
            print(f"Go download the dataset from {self.doi}. (Or remount the external drive on Tycho)")
            raise FileNotFoundError()

        iscell = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'iscell.npy')
        F_all = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'F.npy')
        self.F_all = F_all
        n_cells = int(sum(iscell[:, 0]))

        stat = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'stat.npy', allow_pickle=True)
        ops = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()

        def make_beh(fpath):
            pre_beh = pd.read_csv(fpath)
            columns = ["t"] + list(map(lambda a: f"{a[0]}_{a[1]}", zip(pre_beh.iloc[0, 1:], pre_beh.iloc[1, 1:])))
            columns = {pre_beh.columns[i]: columns[i] for i in range(len(columns))}
            beh = pre_beh.rename(columns=columns).iloc[2:].astype(float).reset_index(drop=True)
            beh.t = beh.t / self.neural_Fs
            return beh

        part, total = self.sub_datset_info.part_of_F[self.sub_dataset]
        block_length = F_all.shape[1] // total

        F_all = F_all - F_all.min(axis=1, keepdims=True)
        # F_all = F_all / np.median(F_all, axis=1, keepdims=True)

        F_all_0 = np.median(F_all, axis=1, keepdims=True)
        F_all = (F_all - F_all_0) / F_all_0

        F_all[np.isnan(F_all)] = 0

        F = F_all[:, (part - 1) * block_length: part * block_length]
        img = Image.open(sub_dataset_base_path / self.sub_datset_info.raw_frames[self.sub_dataset])
        video = None
        if isinstance(video_filename:=self.sub_datset_info.behavior_video[self.sub_dataset], str):
            video = pims.Video(sub_dataset_base_path / video_filename)
        beh = make_beh(sub_dataset_base_path / self.sub_datset_info.behavior_csv[self.sub_dataset])

        nose = self.get_behavior_trace(beh, 'nose')
        body = self.get_behavior_trace(beh, 'bodycenter')
        head = self.get_behavior_trace(beh, 'mouse')

        beh['hd'] = np.arctan2(*(nose - head).T)
        beh['h2b'] = np.linalg.norm(head - body, axis=1)
        beh['x'] = head[:,0]
        beh['y'] = head[:,1]


        return F, img, video, beh, n_cells, stat, ops

    def show_stim_pattern(self, ax, desired_stim):
        ax.matshow(self.ops['meanImg'], cmap='Grays')
        xs, ys = list(zip(*[cell['med'] for cell in self.stat]))
        map = ax.scatter(ys, xs, s=7, c=desired_stim)

        ax.get_figure().colorbar(map)

    @staticmethod
    def get_behavior_trace(beh, point_str, threshold=.999):
        point_trace = np.array([beh.loc[:, point_str + '_x'].to_numpy(),
                                beh.loc[:, point_str + '_y'].to_numpy()]).T
        s = beh.loc[:, point_str + '_likelihood'].to_numpy() < threshold
        point_trace[s] *= np.nan
        return point_trace


class Daie21Dataset:
    dataset_base_path = DATA_BASE_PATH / 'daie21'
    def __init__(self):
        self.neural_data, self.stimuli = self.acquire()

    def acquire(self):

        f = h5py.File(self.dataset_base_path / 'Daie_et_al_2020_targeted_photostim.mat')

        n_sessions = f['data']['dt_si'].shape[0]
        session_n = 0
        dt = f[f['data']['dt_si'][session_n, 0]][0, 0]

        rows = []

        for l_or_r in 'LR':
            n_stim_groups = f[f['data'][l_or_r][session_n, 0]].shape[0]
            for stim_group_n in range(n_stim_groups):
                data = f[f[f['data'][l_or_r][session_n, 0]][stim_group_n, 0]][:]

                if stim_group_n == 0:
                    stim_start, stim_end = np.nan, np.nan
                else:
                    stim_start, stim_end = f[f[f['data']['epochs'][session_n, 0]]['stim'][stim_group_n - 1, 0]][:, 0]

                accuracies = f[f[f['data']['C' + l_or_r][session_n, 0]][stim_group_n, 0]][:, 0]
                for trial, accuracy in zip(data, accuracies):
                    rows.append(dict(l_or_r=l_or_r, stim_group_n=stim_group_n, trial=trial.T, stim_start=stim_start,
                                     stim_end=stim_end, accuracy=accuracy))
        _, n_neurons, n_timepoints = data.shape
        t = np.arange(n_timepoints) * dt

        df = pd.DataFrame(rows)
        nan_rows = df[df['stim_start'].isna()].sample(frac=1).reset_index(drop=True)
        non_nan_rows = df[df['stim_start'].notna()].sample(frac=1).reset_index(drop=True)
        df = pd.concat([nan_rows, non_nan_rows], ignore_index=True)

        def concat_rows(df):
            trials = []
            stims = []
            row_count = 0
            for idx, row in df.iterrows():
                trial = row['trial']
                if np.var(trial) < .1:
                    continue
                stim_start = row['stim_start']
                if not np.isnan(stim_start):
                    group_vec = np.zeros(n_stim_groups - 1)
                    group_vec[row['stim_group_n'] - 1] = 1
                    stims.append(ArrayWithTime(group_vec, (stim_start // dt + 1) * dt + row_count * dt))

                trials.append(trial)
                row_count += trial.shape[0]
                trials.append(np.empty([5, n_neurons]) * np.nan)
                row_count += trials[-1].shape[0]

            A = np.vstack(trials)
            A = ArrayWithTime(A, np.arange(A.shape[0]) * dt)
            stims = ArrayWithTime.from_list(stims)
            return A, stims

        A, stims = concat_rows(df)
        stims.t = stims.t - dt / 50
        return A, stims
