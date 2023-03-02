import numpy as np
import torch
from scipy.integrate import solve_ivp
from typing import Dict, List
from numpy.typing import ArrayLike

data_type = torch.float32

def make_xy_grid(space_lim, grid_pts, to_torch=True):
    X = np.linspace(-space_lim, space_lim, grid_pts)
    Y = np.linspace(-space_lim, space_lim, grid_pts)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_vec = X_grid.reshape([-1])
    Y_vec = Y_grid.reshape([-1])
    XY = np.concatenate([X_vec, Y_vec])[:, None]
    if to_torch:
        XY = torch.from_numpy(XY)
    return XY, (X_grid, Y_grid)


def solve_ode(vel_field, f_u, f_v, init_pos, time_horizon, time_step, space_lim):
    sol = solve_ivp(
        vel_field,
        np.array([0, time_horizon]),
        init_pos,
        t_eval=np.linspace(0, time_horizon, time_step),
    )
    drifter = sol.y
    path = np.concatenate((drifter[0], drifter[1]))
    path = np.reshape(np.concatenate((drifter[0], drifter[1])), [2, len(path) // 2]).T

    Us, Vs = [], []
    for i in path:
        if (
            i[0] >= -space_lim
            and i[1] >= -space_lim
            and i[0] <= space_lim
            and i[1] <= space_lim
        ):
            Us.append(float(f_u(i[0], i[1])))
            Vs.append(float(f_v(i[0], i[1])))

    length = len(Us)
    drifter = [drifter[0][:length], drifter[1][:length]]

    return (
        drifter,
        torch.from_numpy(path[:length]),
        torch.tensor(Us)[:, None],
        torch.tensor(Vs)[:, None],
    )


def downsample(x_grid, y_grid, downsample_factor):
    x_grid = x_grid[::downsample_factor, ::downsample_factor]
    y_grid = y_grid[::downsample_factor, ::downsample_factor]
    x_vec = torch.from_numpy(x_grid.ravel())[:, None]
    y_vec = torch.from_numpy(y_grid.ravel())[:, None]
    return (x_grid, y_grid), (x_vec, y_vec)

def dict_to_np(dictionary):
    np_dict = dict()
    for key, val in dictionary.items():
        new_val = val.numpy() if isinstance(val, torch.Tensor) else val
        np_dict[key] = new_val

    return np_dict

def _to_torch(dictionary: Dict) -> Dict:
    """
    Given a dictionary containing values that may be numpy or torch arrays
    return a new dictionary such that all values are torch arrays with the same
    data type
    """
    new_dict = dict()
    for k, v in dictionary.items():
        if isinstance(v, np.ndarray):
            new_v = torch.from_numpy(v)
            if new_v.dtype in [
                torch.float64,
                torch.float32,
            ]:  # if float cast so types match
                new_v = new_v.double() if data_type == torch.float64 else new_v.float()
        else:
            new_v = v
        new_dict[k] = new_v
    return new_dict


def _to_numpy(dictionary: Dict) -> Dict:
    """
    Convert all values in a dictionary that are torch tensors to numpy arrays
    """
    new_dict = dict()
    for k, v in dictionary.items():
        new_v = v.detach().numpy() if isinstance(v, torch.Tensor) else v
        new_dict[k] = new_v
    return new_dict


def _list_to_numpy(lst: List[torch.Tensor]) -> List[ArrayLike]:
    return [l.detach().numpy() for l in lst]
