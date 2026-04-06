from __future__ import annotations

"""Core two-loop fluxoid reduced model and sector eigensystem cache."""

from dataclasses import dataclass
from typing import Any, cast, overload

import numpy as np
from numpy.typing import NDArray
import scqubits as scq  # type: ignore

from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator
from ztpcraft.bosonic.oscillator_integrals.normal_modes import (
    diagonalize_quadratic_hamiltonian,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FluxoidModelParams:
    """Circuit and flux parameters for the reduced two-loop model."""
    EL_a: float
    EL_b: float
    EJ: float
    EC: float
    phi_ext_a: float
    phi_ext_b: float
    flux_allocation_alpha: float


@dataclass(frozen=True)
class FluxoidSector:
    """Fluxoid sector label `(m_a, m_b)`."""
    m_a: int
    m_b: int


@dataclass
class FluxoidSectorState:
    """Cached eigensystem and oscillator metadata for one sector."""
    sector: FluxoidSector
    evals: FloatArray
    evecs: ComplexArray

    osc: LocalHarmonicOscillator
    shift: float
    cutoff: int


class TwoLoopFluxoidModel:
    """Reduced single-mode model parameterized by fluxoid sector."""

    def __init__(self, params: FluxoidModelParams):
        """Initialize model with physical and flux-allocation parameters.

        Parameters
        ----------
        params:
            Physical circuit parameters and external flux settings.
        """
        self.params = params

    def inductive_fractions(self) -> tuple[float, float, float]:
        """Return inductive-energy sum and fractional weights.

        Returns
        -------
        tuple[float, float, float]
            `(EL_sum, f_a, f_b)` where `f_a = EL_a / EL_sum` and
            `f_b = EL_b / EL_sum`.
        """
        EL_a = self.params.EL_a
        EL_b = self.params.EL_b
        EL_sum = EL_a + EL_b
        return EL_sum, EL_a / EL_sum, EL_b / EL_sum

    def effective_flux(self, sector: FluxoidSector) -> float:
        """Effective external flux seen by the reduced 1D model.

        Parameters
        ----------
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        float
            Effective reduced-model external flux value.
        """
        p = self.params
        _, f_a, f_b = self.inductive_fractions()
        return -f_a * (p.phi_ext_a + 2.0 * np.pi * sector.m_a) + f_b * (
            p.phi_ext_b + 2.0 * np.pi * sector.m_b
        )

    def get_sector_key(self, sector: FluxoidSector) -> tuple[int, int]:
        """Return dictionary/cache key for a sector.

        Parameters
        ----------
        sector:
            Fluxoid sector label.

        Returns
        -------
        tuple[int, int]
            Cache key `(m_a, m_b)`.
        """
        return (sector.m_a, sector.m_b)

    def get_f1_f2(self, sector: FluxoidSector) -> tuple[float, float]:
        """Split effective flux into allocation components.

        Parameters
        ----------
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        tuple[float, float]
            `(f1, f2)` according to `flux_allocation_alpha`.
        """
        phi_eff = self.effective_flux(sector)
        f1 = self.params.flux_allocation_alpha * phi_eff
        f2 = (1.0 - self.params.flux_allocation_alpha) * phi_eff
        return f1, f2

    def coordinate_shift(self, sector: FluxoidSector) -> float:
        """Return oscillator-center shift used for overlap calculations.

        Parameters
        ----------
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        float
            Harmonic-oscillator center shift.
        """
        f1, _ = self.get_f1_f2(sector)
        return f1

    def energy_offset(self, sector: FluxoidSector) -> float:
        """Energy offset from the fluxoid number model.

        Parameters
        ----------
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        float
            Sector-dependent additive energy offset.
        """
        p = self.params
        EL_sum, f_a, f_b = self.inductive_fractions()
        phi_a = p.phi_ext_a + 2.0 * np.pi * sector.m_a
        phi_b = p.phi_ext_b + 2.0 * np.pi * sector.m_b
        return (
            0.5
            * EL_sum
            * (f_a * phi_a**2 + f_b * phi_b**2 - (f_a * phi_a - f_b * phi_b) ** 2)
        )

    @overload
    def potential_as_explicit_inductor_sum(
        self, phi: float, sector: FluxoidSector
    ) -> float: ...

    @overload
    def potential_as_explicit_inductor_sum(
        self, phi: FloatArray, sector: FluxoidSector
    ) -> FloatArray: ...

    def potential_as_explicit_inductor_sum(
        self, phi: float | FloatArray, sector: FluxoidSector
    ) -> float | FloatArray:
        """
        Full two-loop reduced potential:
        U(phi) = EL_a/2 (phi - phi_ext_a - 2π m_a)^2
               + EL_b/2 (phi + phi_ext_b + 2π m_b)^2
               - EJ cos(phi)

        Parameters
        ----------
        phi:
            Phase coordinate (scalar or array).
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        float | FloatArray
            Potential value(s) at `phi`.
        """
        p = self.params
        if np.isscalar(phi):
            phi_scalar = cast(float, phi)
            inductor_1_flux = phi_scalar - p.phi_ext_a - 2.0 * np.pi * sector.m_a
            inductor_2_flux = phi_scalar + p.phi_ext_b + 2.0 * np.pi * sector.m_b
            return (
                0.5 * p.EL_a * inductor_1_flux**2
                + 0.5 * p.EL_b * inductor_2_flux**2
                - p.EJ * np.cos(phi_scalar)
            )

        phi_array = np.asarray(phi, dtype=np.float64)
        inductor_1_flux = phi_array - p.phi_ext_a - 2.0 * np.pi * sector.m_a
        inductor_2_flux = phi_array + p.phi_ext_b + 2.0 * np.pi * sector.m_b
        return cast(
            FloatArray,
            0.5 * p.EL_a * inductor_1_flux**2
            + 0.5 * p.EL_b * inductor_2_flux**2
            - p.EJ * np.cos(phi_array),
        )

    @overload
    def potential_as_effective_inductor_plus_offset(
        self, phi: float, sector: FluxoidSector
    ) -> float: ...

    @overload
    def potential_as_effective_inductor_plus_offset(
        self, phi: FloatArray, sector: FluxoidSector
    ) -> FloatArray: ...

    def potential_as_effective_inductor_plus_offset(
        self, phi: float | FloatArray, sector: FluxoidSector
    ) -> float | FloatArray:
        """
        Equivalent single-inductor form:
        U(phi) = EL/2 * (phi + phi_eff)^2 - EJ cos(phi) + offset

        Parameters
        ----------
        phi:
            Phase coordinate (scalar or array).
        sector:
            Fluxoid sector `(m_a, m_b)`.

        Returns
        -------
        float | FloatArray
            Potential value(s) at `phi`.
        """
        p = self.params
        EL_sum, _, _ = self.inductive_fractions()
        phi_eff = self.effective_flux(sector)
        offset = self.energy_offset(sector)

        if np.isscalar(phi):
            phi_scalar = cast(float, phi)
            return (
                0.5 * EL_sum * (phi_scalar + phi_eff) ** 2
                - p.EJ * np.cos(phi_scalar)
                + offset
            )

        phi_array = np.asarray(phi, dtype=np.float64)
        return cast(
            FloatArray,
            0.5 * EL_sum * (phi_array + phi_eff) ** 2
            - p.EJ * np.cos(phi_array)
            + offset,
        )

    def get_scq_fluxonium_object(
        self, sector: FluxoidSector, cutoff: int = 110
    ) -> tuple[Any, float, LocalHarmonicOscillator]:
        """
        Construct scqubits Fluxonium for this sector.

        IMPORTANT:
        - scq uses flux in units of Phi0 (i.e. 2π = 1 flux quantum)

        Parameters
        ----------
        sector:
            Fluxoid sector `(m_a, m_b)`.
        cutoff:
            Basis cutoff used to build the scqubits object.

        Returns
        -------
        tuple[Any, float, LocalHarmonicOscillator]
            `(scq_fluxonium, coordinate_shift, local_oscillator)`.
        """
        p = self.params
        EL_sum, _, _ = self.inductive_fractions()
        phi_eff = self.effective_flux(sector)
        flux = phi_eff / (2.0 * np.pi)

        scq_fluxonium = scq.Fluxonium(  # type: ignore
            EJ=p.EJ,
            EC=p.EC,
            EL=EL_sum,
            flux=flux,
            cutoff=cutoff,
        )
        shift = self.coordinate_shift(sector)

        nm = diagonalize_quadratic_hamiltonian(np.array([[p.EC]]), np.array([[EL_sum]]))
        osc = LocalHarmonicOscillator(
            transform_xi_theta=nm.transform_xi_theta,
            transform_qn=nm.transform_qn,
            frequencies=nm.frequencies,
            center=np.array([shift]),
        )
        return scq_fluxonium, shift, osc  # type: ignore


class FluxoidSectorManager:
    """Lazy eigensystem cache over sectors for a fixed model/cutoff."""

    def __init__(
        self,
        model: TwoLoopFluxoidModel,
        cutoff: int = 110,
        evals_count: int = 20,
    ):
        """Create manager with fixed diagonalization settings.

        Parameters
        ----------
        model:
            Two-loop reduced model instance.
        cutoff:
            Basis cutoff for sector diagonalization.
        evals_count:
            Number of eigenpairs retained per sector.
        """
        self.model = model
        self.cutoff = cutoff
        self.evals_count = evals_count

        self._cache: dict[tuple[int, int], FluxoidSectorState] = {}

    def _key(self, sector: FluxoidSector) -> tuple[int, int]:
        """Internal cache key helper.

        Parameters
        ----------
        sector:
            Fluxoid sector label.

        Returns
        -------
        tuple[int, int]
            Cache key `(m_a, m_b)`.
        """
        return (sector.m_a, sector.m_b)

    def get_sector_state(self, sector: FluxoidSector) -> FluxoidSectorState:
        """Get (or build and cache) eigensystem data for `sector`.

        Parameters
        ----------
        sector:
            Fluxoid sector to retrieve.

        Returns
        -------
        FluxoidSectorState
            Eigensystem and overlap-metadata object for the requested sector.
        """
        key = self._key(sector)

        if key in self._cache:
            return self._cache[key]

        scq_obj, shift, osc = self.model.get_scq_fluxonium_object(
            sector, cutoff=self.cutoff
        )

        evals_raw, evecs_raw = scq_obj.eigensys(evals_count=self.evals_count)
        evals = np.asarray(evals_raw, dtype=np.float64)
        evecs = np.asarray(evecs_raw, dtype=np.complex128)

        state = FluxoidSectorState(
            sector=sector,
            evals=evals,
            evecs=evecs,
            shift=shift,
            cutoff=self.cutoff,
            osc=osc,
        )

        self._cache[key] = state
        return state
