import os

import numpy as np
import pandas
import yaml
from scipy import sparse
from yaml.loader import SafeLoader

__all__ = ["PocketModel"]

DEFAULT_BACKEND = "numpy-nr"


#
# config | may move in a config.py
#

_SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
CORRECTION_FILEPATH = os.path.join(_SOURCEDIR, "data", "pocket_corrections.yaml")

# CONFIG
with open(CORRECTION_FILEPATH) as f:
    data = yaml.load(f, Loader=SafeLoader)
    POCKET_PARAMETERS = pandas.DataFrame(data["data"]).set_index(["ccdid", "qid"])


def get_config(ccdid, qid):
    """ returns the pocket effect parameter configuration for the given quadrant """
    return POCKET_PARAMETERS.loc[ccdid, qid]

class PocketModel():

    def __init__(self, alpha, cmax, beta, nmax):
        """
        cmax: float
            pocket capacity

        nmax: float
            pixel capacity (not quite the full well)

        alpha: float
            from pocket transfer dynamics

        beta: float
            to-pocket transfer dynamics

        """
        self._alpha = alpha
        self._cmax = cmax
        self._beta = beta
        self._nmax = nmax


    # ============= #
    #   Top level   #
    # ============= #


    # ============= #
    #  Model func   #
    # ============= #
    def flush(self, pocket_q):
        """  transfer of electrons from the pocket to the pixels.

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        Returns
        -------
        float, Array
            charge leaving the pocket.
        """

    def get_delta(self, pocket_q, pixel_q):
        """ net pocket charge transfert

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        pixel_q: float, Array
            pixel charge prior read-out (undistorted)

        Returns
        -------
        float, Array
            pixel charge excess (>0) and deficit (<0) at the read-out.
        """
        # flush
        x = pocket_q / self._cmax
        from_pocket = np.clip(self._cmax * x**self._alpha, 0, pocket_q)

        # fill
        y = pixel_q / self._nmax
        to_pocket = np.clip(self._cmax * (1 - x)**self._alpha * y**self._beta,  0., pixel_q)

        delta = from_pocket - to_pocket
        return delta

    def get_pocket_and_corr(self, pocket_q, pixel_q):
        """ scanning function providing corrected pixel and new pocket charge

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        pixel_q: float, Array
            pixel charge prior read-out (undistorted)

        Returns
        -------
        list
            - new pocket charge: float, Array
            - corrected pixel: float, Array
        """
        delta = self.get_delta(pocket_q, pixel_q)
        pixel_corr = pixel_q + delta
        new_pocket = pocket_q - delta
        return new_pocket, pixel_corr

    def apply(self, pixels, init=None, backend=DEFAULT_BACKEND):
        """ pocket effect correction

        Parameters
        ----------
        pixels: 2d-Array
            raw pixel map, including overhead of shape (M,N)

        init: None, Array
            initial condition of the pocket (M,).
            If None, zero is assumed.

        backend: str
            backend used for the computation

        Returns
        -------
        2d-Array
            pocket effect on pixel map (M,N)
        """
        # special case, computation not from python
        if backend == "cpp":
            from ._pocket import _PocketModel as PocketModelCPP
            thiscpp = PocketModelCPP(self._alpha, self._cmax, self._beta, self._nmax)
            return thiscpp.apply(pixels) # 0 is force here.

        # good format
        pixels = np.atleast_2d(pixels)
        if init is None:
            init = np.zeros(shape=pixels[:,0].shape)

        # call current sub-function
        if backend == "jax":
            return self._scan_apply(pixels, init=init)

        elif backend == "numpy":
            return self._forloop_apply(pixels, init=init)

        elif backend == "numpy-nr":
            return self._forloop_apply_baseline(pixels, init=init)

        else:
            raise ValueError(f"unknown backend {backend}")

    def get_sparse_hessian(self, test_column, backend=DEFAULT_BACKEND):
        """ """
        jacobian = self.get_jacobian(test_column, backend=backend)

        i, j = np.meshgrid(np.arange(jacobian.shape[0]),
                           np.arange(jacobian.shape[1]))
        i, j = i.flatten(), j.flatten() # flattend
        v = jacobian[i.flatten(), j.flatten()]
        non_zero_idx = np.abs(v)>1.E-5

        jac_sparse = sparse.coo_matrix(( v[non_zero_idx],
                                         (i[non_zero_idx], j[non_zero_idx])
                                       ), shape=jacobian.shape)

        hessian_sparse = jac_sparse.T @ jac_sparse
        return hessian_sparse

    def get_jacobian(self, test_column, backend=DEFAULT_BACKEND):
        """ """
        # to be moved inside class
        jacobian = pocket_model_derivatives(self, test_column, backend=DEFAULT_BACKEND)
        return jacobian

    # ====================== #
    # apply backend supports #
    # ====================== #
    def _scan_apply(self, pixels, init=None):
        """ docstring, see: self.apply """
        # with for lax.scan | jax
        # atleast_2d and squeeze is to respect cpp-version behavior
        import jax
        last_pocket, resbuff = jax.lax.scan(self.get_pocket_and_corr,
                                                init,
                                                np.ascontiguousarray(pixels.T))
        return resbuff.T.squeeze()

    def _forloop_apply(self, pixels, init):
        """ docstring, see: self.apply """
        # with for loop | numpy
        pocket = init # for consistency between method
        resbuff = []

        for col in pixels.T:
            pocket, corr = self.get_pocket_and_corr(pocket, col)
            resbuff.append(corr) # build line by line

        return np.vstack(resbuff).T

    def _forloop_apply_baseline(self, pix, init):
        """apply the model to 2D image

        = original NR dev =

        Parameters
        ----------
        pix : 2D array-like of floats
          we assume that i labels the rows and j labels the physical columns.

        .. note:: columns and rows are *not* interchangeable here !
        """
        nrows, ncols = pix.shape
        pocket = init # for consistency between method
        pix = np.ascontiguousarray(pix.T)

        cmax_inv = 1 / self._cmax
        pix_beta = pix / self._nmax

        # make sure input data does not contain negative values, so we
        # don't get NaNs when computing the power
        np.maximum(pix_beta, 0, out=pix_beta)
        np.power(pix_beta, self._beta, out=pix_beta)

        for j in range(ncols):
            # from_pocket = self.flush(pocket):
            # c_max * (pocket / c_max)**alpha
            tmp = pocket * cmax_inv
            from_pocket = tmp**self._alpha
            from_pocket *= self._cmax
            np.clip(from_pocket, 0, pocket, out=from_pocket)

            # to_pocket = self.fill(pocket, n_j):
            # cmax * (1 - pocket / cmax)**alpha * (pixel / nmax)**beta
            to_pocket = 1 - tmp
            np.power(to_pocket, self._alpha, out=to_pocket)
            to_pocket *= pix_beta[j] * self._cmax
            np.clip(to_pocket,  0.,  pix[j], out=to_pocket)

            delta = from_pocket - to_pocket

            pix[j] += delta
            pocket -= delta

            # just making sure that the pocket contents never become negative
            np.maximum(pocket, 0, out=pocket)

        return pix.T



def pocket_model_derivatives(model, pix, step=0.01, backend=DEFAULT_BACKEND):
    """model derivatives w.r.t the pixel values

    For now, we use numerical derivatives. It is probably possible to do better.

    Parameters
    ----------
    model : PocketModel
      the pocket effect model
    pix : array_like
      the pixel array
    step : float
      numerical step

    Returns
    -------
    jacobian matrix : array_like
    """
    N = len(pix)
    pixim = np.resize(pix, (N+1, N))
    np.fill_diagonal(pixim, pixim.diagonal() + step)
    vv = model.apply(pixim, backend=backend)

    if backend == "cpp":
        v0 = model.apply(pix, backend=backend)
    else:
        v0 = vv[-1]

    J = (vv[:-1] - v0) / step
    return np.triu(J)
