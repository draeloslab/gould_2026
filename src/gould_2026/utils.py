import numpy as np
import warnings


def is_orthonormal(Q, rows_too=False):
    o = np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
    if rows_too:
        o = o and np.allclose(Q @ Q.T, np.eye(Q.shape[0]))
    return o

def principle_angles(Q1, Q2):
    assert is_orthonormal(Q1) and is_orthonormal(Q2)
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    return np.arccos(np.clip(s, -1, 1))

def align_column_spaces(A, B):
    # https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, (B).T


def column_space_distance(Q1, Q2, method='grassmann', override_ortho_check=False):
    if not override_ortho_check:
        for Q in Q1, Q2:
            assert is_orthonormal(Q)
    else:
        warnings.warn('this method is intended to be used for only orthogonal matrices')

    if method == 'grassmann':
        return np.linalg.norm(principle_angles(Q1, Q2))
    elif method == 'angles' or method == 'nuclear':
        return np.abs(principle_angles(Q1, Q2)).sum()
    elif method == 'aligned_diff':
        Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
        return np.linalg.norm(Q1_rotated - Q2)
    else:
        raise ValueError()

def angle_between(v1, v2, radians=False, subspace=False):
    v1_u = v1.flatten() / np.linalg.norm(v1)
    v2_u = v2.flatten() / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if subspace:
        angle = min(angle, np.pi - angle)
    return angle * (180.0 / np.pi if not radians else 1.0)

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

def check_same(v, var_name='temp', overwrite=True):
    """
    >>> check_same(1) # doctest: +ELLIPSIS
    temp: ...
    >>> check_same(1)
    temp: True
    >>> check_same(2)
    temp: False
    """
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
    except ImportError:
        pass
    s = f'/tmp/_{var_name}'
    try:
        old_v = np.load(f"{s}.npy")
        assert old_v is not None # edge case I don't want to deal with
    except FileNotFoundError:
        old_v = None

    if overwrite:
        np.save(s, v)

    if old_v is not None:
        same = np.shape(v) == np.shape(old_v) and np.nanmax((v - old_v) ** 2) == 0
        print(f'{var_name}: {same}')
    else:
        print(f'{var_name}: NEW')
