from ninatool.internal.structures import branch
from ninatool.internal.elements import J
import numpy as np
import scipy as sp
from typing import List, Tuple


class FlxnArrayMode:
    def __init__(
        self,
        EJa: float,
        EJb: float,
        N_junctions: int,
        Cga: float,
        CJa: float,
        CJb: float,
        phi_ext: float,
        N_flux_points_for_minima: int = 1000,
        autorun: bool = True,
    ):
        """ 
        Class for analyzing array modes of a single-loop fluxonium circuit.

        Parameters
        ----------
        EJa : float
            The Josephson energy of the array junctions.
        EJb : float
            The Josephson energy of the bias loop.
        N_junctions : int
            The number of junctions in the array.
        Cga : float
            The capacitance of the array junctions.
        CJa : float
            The capacitance of the bias loop.
        CJb : float
            The capacitance of the bias loop.
        phi_ext : float
            The external flux in the bias loop.
        N_flux_points_for_minima : int, optional
            The number of flux points to use for the minima finding. Defaults to 1000.
        autorun : bool, optional
            Whether to run the minima finding automatically. Defaults to True.
        """ 
        self.EJa: float = EJa
        self.EJb: float = EJb
        self.N_junctions: int = N_junctions
        self.Cga: float = Cga
        self.CJa: float = CJa
        self.CJb: float = CJb
        self.phi_ext: float = phi_ext
        self.flxn_nina_branch: "branch" = None

        if autorun:
            self.flxn_nina_branch = self.generate_flxn_nina_branch_for_minima(
                EJa, EJb, N_junctions, N_flux_points_for_minima
            )

    def run_minima_finding(self):
        pass

    @staticmethod
    def generate_flxn_nina_branch_for_minima(
        EJa: float, EJb: float, N_junctions: int, N_flux_points: int
    ) -> "branch":
        flxn_circuit_element_list = [J(EJa, name=str(i)) for i in range(N_junctions)]
        bs_junction = J(EJb, name="bs")
        flxn_circuit_element_list.append(bs_junction)
        flxn_nina_branch = branch(flxn_circuit_element_list, name="array")
        flxn_nina_branch.free_phi = np.linspace(-np.pi, np.pi, N_flux_points)
        return flxn_nina_branch

    @staticmethod
    def extract_branch_phi_min_max(flxn_nina_branch: "branch") -> Tuple[float, float]:
        return np.min(flxn_nina_branch.phi), np.max(flxn_nina_branch.phi)

    @staticmethod
    def generate_phi_ext_alias(
        phi_ext: float, branch_phi_min: float, branch_phi_max: float
    ) -> List[float]:
        return np.concatenate(
            [
                np.arange(phi_ext - 2 * np.pi, branch_phi_min, -2 * np.pi),
                np.arange(phi_ext, branch_phi_max, 2 * np.pi),
            ]
        ).sort()

    @staticmethod
    def hessian_matrix(
        EJa: float,
        EJb: float,
        N_junctions: int,
        array_junction_flux: float,
        external_flux: float,
    ) -> np.ndarray:
        hessian = (
            EJb
            * np.ones((N_junctions, N_junctions))
            * np.cos(-N_junctions * array_junction_flux + external_flux)
        )
        hessian += EJa * np.cos(array_junction_flux) * np.identity(N_junctions)
        return hessian

    @staticmethod
    def find_optimal_array_junction_phi(
        flxn: "branch", phi_ext_alias: List[float]
    ) -> np.ndarray:
        # make splines
        branch_phi_spline = sp.interpolate.CubicSpline(
            flxn.free_phi, flxn.phi, extrapolate=False
        )
        array_phi_spline = sp.interpolate.CubicSpline(
            flxn.free_phi, flxn.elements[0].phi, extrapolate=False
        )
        # obtain all the roots of the spline, by taking offsets of phi_ext_alias
        optimal_free_phi = []
        optimal_array_junction_phi = []
        for phi_ext_offset in phi_ext_alias:
            pp_shifted = sp.interpolate.PPoly(
                branch_phi_spline.c.copy(), branch_phi_spline.x, extrapolate=False
            )
            pp_shifted.c[-1, :] -= phi_ext_offset
            optimal_free_phi.extend(pp_shifted.roots())
        optimal_free_phi = np.array(optimal_free_phi)
        optimal_free_phi.sort()
        optimal_array_junction_phi = array_phi_spline(optimal_free_phi)
        return optimal_array_junction_phi

    @staticmethod
    def sieve_out_minima(
        optimal_array_junction_phi: np.ndarray,
        phi_ext: float,
        EJa: float,
        EJb: float,
        N_junctions: int,
    ) -> List[float]:
        minima_array_junction_phi = []
        for idx in range(len(optimal_array_junction_phi)):
            evals_hessian_candidate = np.linalg.eigvals(
                FlxnArrayMode.hessian_matrix(
                    EJa, EJb, N_junctions, optimal_array_junction_phi[idx], phi_ext
                )
            )
            if np.all(evals_hessian_candidate.real > 0):
                minima_array_junction_phi.append(optimal_array_junction_phi[idx])
        return minima_array_junction_phi
