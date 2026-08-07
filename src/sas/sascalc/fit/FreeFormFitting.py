"""
FreeFormFitting module runs free-form SAS inversion via the external `ffsi`
package (GALAHAD SNLS solver). Basic showcase: sphere model, 1D data.

This is a sibling of BumpsFitting: a FitEngine subclass returning FResult
objects through the same FitThread pipeline, so the GUI drives it exactly
like a bumps fit. The optimizer is not bumps -- ffsi solves for
the full radius distribution (bin weights on a simplex, smoothness-
regularized) in a single constrained solve.
"""
import logging
from dataclasses import dataclass

import numpy as np

from sas.sascalc.fit.AbstractFitEngine import FitEngine, FResult

logger = logging.getLogger(__name__)

try:
    from ffsi.models import Sphere
    from ffsi.optimize_galahad import optimize as ffsi_optimize
    from ffsi.utils import contract_tensor
    FFSI_AVAILABLE = True
except ImportError as exc:
    FFSI_AVAILABLE = False
    _FFSI_IMPORT_ERROR = str(exc)
    logger.info("ffsi not available, free-form inversion disabled: %s", exc)

DEFAULT_SIGMA = 0.25

# Free-form is a sphere-only for now for the
# basic showcase of the free-form sas inversion.
SUPPORTED_MODEL = "sphere"
FREE_FORM_FIT_PARAMS = ["scale", "background"]


def model_id(model):
    """The sasmodels id of a SasviewModel, falling back to its name."""
    return getattr(model, "id", model.name)


def is_supported(model):
    """True if free-form inversion can handle this model (sphere only)"""
    return model_id(model) == SUPPORTED_MODEL


@dataclass
class FreeFormResult:
    """Output of one sphere inversion."""
    scale: float          # SasView sphere scale (volume fraction), converted from xi
    xi: float             # raw ffsi scale: I_opt = xi * Gw + background
    background: float     # b_opt
    r: np.ndarray         # radius bin centers
    w: np.ndarray         # radius distribution weights
    q: np.ndarray         # masked q the inversion ran on
    theory: np.ndarray    # I_opt = xi * Gw + background on q


def invert_sphere(q, iq, diq, r_min, r_max, n_bins, sigma=DEFAULT_SIGMA, drho=1.0):
    """
    Free-form sphere inversion: the body of ffsi's
    sas_inversion_sphere_real_data.py minus file I/O and plotting.

    ffsi convention drho = 1
    """
    if not FFSI_AVAILABLE:
        raise RuntimeError("ffsi is not installed: %s" % _FFSI_IMPORT_ERROR)

    q = np.ascontiguousarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)
    diq = np.asarray(diq, dtype=float)
    r = np.linspace(float(r_min), float(r_max), int(n_bins))

    # Solve with the ffsi convention drho = 1
    # TODO: generalise invert_shape beyond the sphere
    green = Sphere.compute_scattering_intensity([q], [r], 1.0)
    xi, background, w_list = ffsi_optimize(green, iq, diq, sigma=sigma)
    w = np.asarray(w_list[0])

    theory = float(xi) * np.asarray(contract_tensor(green, [w], skip_axes=[0])) + float(background)

    # Convert ffsi's raw xi to a SasView sphere scale (volume fraction).
    #   scale = xi * <V> * 1e4 / drho^2,  <V> = sum_k w_k (4/3 pi r_k^3).
    avg_volume = float(Sphere.compute_average_volume([r], [w]))
    scale = float(xi) * avg_volume * 1e4 / float(drho) ** 2

    return FreeFormResult(scale=scale, xi=float(xi), background=float(background),
                          r=r, w=w, q=q, theory=theory)


class FreeFormFit(FitEngine):
    """
    Fit a radius distribution to the data using free-form inversion (ffsi).

    Driven through the same interface as BumpsFit: set_model()/set_data()
    from the FitEngine base class, then fit() from a FitThread.
    """
    def __init__(self, bins=None, sigma=DEFAULT_SIGMA):
        """
        :param bins: dict {parameter name: (min, max, nbins)}; needs 'radius'
        :param sigma: smoothness regularization weight
        """
        FitEngine.__init__(self)
        self.fitter_id = None
        self.bins = dict(bins) if bins else {}
        self.sigma = sigma

    def fit(self, msg_q=None, q=None, handler=None, curr_thread=None,
            ftol=1.49012e-8, reset_flag=False):
        # ftol/reset_flag are part of the FitThread calling convention but
        # have no meaning for the GALAHAD solve.
        if handler is not None:
            # handler.error() stringifies the last result, so seed one
            handler.set_result("Free-form inversion (GALAHAD SNLS)")

        all_results = []
        for fit_id, arrange in self.fit_arrange_dict.items():
            if not arrange.get_to_fit():
                continue
            if curr_thread is not None:
                curr_thread.isquit()   # raises KeyboardInterrupt on stop

            model = arrange.get_model().model     # the SasviewModel
            if not is_supported(model):
                raise ValueError("Free-form inversion only supports the %s model." % SUPPORTED_MODEL)
            if 'radius' not in self.bins:
                raise ValueError("No bin specification for parameter 'radius'.")

            # Contrast for converting ffsi's xi into a SasView volume-fraction scale
            # only used to report the correct scaled Scale to the model pane.
            drho = model.getParam('sld') - model.getParam('sld_solvent')

            fitdata = arrange.get_data()          # FitData1D
            idx = fitdata.idx
            r_min, r_max, n_bins = self.bins['radius']
            result = invert_sphere(fitdata.x[idx], fitdata.y[idx], fitdata.dy[idx],
                                   r_min, r_max, n_bins, sigma=self.sigma, drho=drho)

            fitting_result = FResult(model=model, data=fitdata,
                              param_list=list(FREE_FORM_FIT_PARAMS))
            fitting_result.theory = result.theory
            fitting_result.residuals = (result.theory - fitdata.y[idx]) / fitdata.dy[idx]
            fitting_result.index = idx
            fitting_result.fitter_id = self.fitter_id
            fitting_result.success = True
            fitting_result.mesg = ''
            fitting_result.pvec = np.array([result.scale, result.background])
            # GALAHAD returns no uncertainties
            fitting_result.stderr = np.zeros(2)
            # chi2 = sum(res**2)/res.size, matching ffsi's reference and SasView's calculateChi2.
            n_pts = max(1, int(np.sum(idx)))
            fitting_result.fitness = float(np.sum(fitting_result.residuals ** 2) / n_pts)
            fitting_result.convergence = np.empty((0, 1), 'd')
            fitting_result.freeform = result
            all_results.append(fitting_result)

        if not all_results:
            raise RuntimeError("Nothing to fit")

        if q is not None:
            q.put(all_results)
            return q
        return all_results
