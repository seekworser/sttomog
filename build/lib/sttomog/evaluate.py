"""
This is the collection of methods for evaluate density matrix.
"""
import numpy as np


__all__ = [
    "fidelity",
    "concurrence",
]


def fidelity(rho, sigma):
    """
    This function returns the fidelity between rho and sigma
    
    Parameters:
    ----------
    rho : numpy.ndarray
        density matrix 1
    sigma : numpy.ndarray
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


def concurrence(rho):
    """
    This function returns the concurrence of rho.

    Parameters:
    ----------
    rho : np.ndarray
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
        dtype=float_type
    )
    tilde_rho = z.dot(self.rho_hv.T.conj().dot(z))
    r = sqrt_rho.dot(tilde_rho.dot(sqrt_rho))
    eig = np.linalg.eig(r)[0]
    tmp = np.sort([np.sqrt(np.max([0, i])) for i in np.real(eig)])
    con = np.real(tmp[3] - tmp[2] - tmp[1] - tmp[0])
    con = np.max([con, 0.])
    return con
