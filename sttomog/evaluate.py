"""
This is the collection of methods for evaluate density matrix.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


__all__ = [
    "fidelity",
    "concurrence",
    "plot_rho",
]

cm_colors = [
    "#1f77b4",
    "#1f77b4"
]
cm_values = range(len(cm_colors))
color_list = []
for v, c in zip(cm_values, cm_colors):
    color_list.append((v / np.max(cm_values), c))
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list)

float_type = np.float
complex_type = np.complex128


class InputError(Exception):
    pass


def fidelity(rho: np.ndarray, sigma: np.ndarray):
    """
    This function returns the fidelity between rho and sigma

    Parameters:
    ----------
    rho : {np.ndarray}
        density matrix 1
    sigma : {np.ndarray}
        density matrix 2

    Returns
    -------
    float
        fidelity between rho and sigma
    """
    [eig, uni] = np.linalg.eig(rho)
    eig = [np.sqrt(max(0, i)) for i in np.real(eig)]
    sqrt_rho = uni.dot(np.diag(eig).dot(uni.T.conj()))
    rho_all = sqrt_rho.dot(sigma.dot(sqrt_rho))
    [eig, uni] = np.linalg.eig(rho_all)
    eig = [np.sqrt(max(0, i)) for i in np.real(eig)]
    sqrt_rho_all = uni.dot(np.diag(eig).dot(uni.T.conj()))
    return float_type(np.real(np.trace(sqrt_rho_all)))


def concurrence(rho: np.ndarray):
    """
    This function returns the concurrence of rho.

    Parameters:
    ----------
    rho : {np.ndarray}
        density matrix for which concurrence is calculated

    Returns
    -------
    float
        concurrence of rho
    """
    [eig, uni] = np.linalg.eig(rho)
    eig = [np.sqrt(max(0, i)) for i in np.real(eig)]
    sqrt_rho = uni.dot(np.diag(eig).dot(uni.T.conj()))
    z = np.array(
        [
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]
        ],
        dtype=complex_type
    )
    tilde_rho = z.dot(rho.T.conj().dot(z))
    r = sqrt_rho.dot(tilde_rho.dot(sqrt_rho))
    eig = np.linalg.eig(r)[0]
    tmp = np.sort([np.sqrt(np.max([0, i])) for i in np.real(eig)])
    con = np.real(tmp[3] - tmp[2] - tmp[1] - tmp[0])
    con = np.max([con, 0.])
    return con


def plot_rho(
    rho: np.ndarray,
    box_size: float =0.6,
    real_imag: str ='real',
    color_style: str ="cmap",
    colormap: mpl.colors.Colormap =custom_cmap
):
    """
    This function plots the real/imaginary part of density matrix.

    Parameters:
    ----------
    rho : {np.ndarray}
        density matrix to be plotted
    box_size : {float}, optional
        box size of bar plot
        (the default is 0.6, which means box size is 0.6)
    real_imag : {str}, optional
        which part of rho to be blotted, real part/imginary part
        (the default is 'real', which means real part of rho to be plotted)
    color_style : {str}, optional
        what color style to be used. avairable value is "cmap" and "bw".
        "cmap" will color the bar depending on its height and given colormap.
        "blue" will color the bar with blue (colormap will be ignored)
        "bw" will color the bar with blue and white (colormap will be ignored)
        (the default is "bw")
    colormap : {mpl.colors.Colormap}, optional
        colormap to be used to color the bars.
        (the default is custom_cmap, which is set to all blue)

    Returns
    -------
    None
    """
    if box_size > 1 or box_size < 0:
        return None
    pos_value = 0.5 * (1 - box_size)

    x_mesh, y_mesh = np.meshgrid(np.arange(4), np.arange(4))
    x_mesh = x_mesh + pos_value
    y_mesh = y_mesh + pos_value
    x_pos = x_mesh.flatten()
    y_pos = y_mesh.flatten()
    z_pos = np.zeros_like(x_pos)
    dx = np.ones_like(x_pos) * box_size
    dy = dx.copy()
    if real_imag == 'real':
        dz = np.real(rho).flatten()
    elif real_imag == "imag":
        dz = np.imag(rho).flatten()
    else:
        raise InputError
    dz_plot = np.zeros_like(dz)
    for i in range(len(dz)):
        if dz[i] < 0:
            z_pos[i] = dz[i]
            dz_plot[i] = -dz[i]
        else:
            dz_plot[i] = dz[i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(-1, 1)
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_xticklabels(['', '', '', ''])
    ax.set_yticklabels(['', '', '', ''])
    ax.set_zticks(np.arange(-1, 1.5, 0.5))
    ax.set_zticklabels(['', '', '', '', ''])
    ax.grid(False)

    values = (dz - dz.min()) / np.float_(1 + 1)
    if color_style == "cmap":
        colors = colormap(values)
    elif color_style == "bw":
        bl = [0.12156863, 0.46666667, 0.70588235, 1.]
        wh = [0.80, 0.80, 0.80, 1.]
        colors = [
            bl, bl, bl, wh,
            bl, bl, bl, wh,
            bl, bl, bl, wh,
            wh, wh, wh, bl
        ]
    elif color_style == "blue":
        pass
    else:
        raise InputError
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz_plot, color=colors)
    return
