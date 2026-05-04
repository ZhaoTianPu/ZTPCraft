from __future__ import annotations

"""Core tunneling-assisted (phase-slip) reduced model and sector eigensystem cache."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import scqubits as scq  # type: ignore

from ztpcraft.bosonic.oscillator_integrals.normal_modes import (
    diagonalize_quadratic_hamiltonian,
)
from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TunnelingAssistedModelParams:
    """Physical parameters for the tunneling-assisted (phase-slip) reduced model."""

    EL_a: float
    EL_b: float
    EJ: float
    EC: float
    phi_ext_a: float
    phi_ext_b: float
    E_S: complex


@dataclass(frozen=True)
class PhaseSlipSector:
    """Phase-slip sector labeled by integer ``p``."""

    p: int


@dataclass
class PhaseSlipSectorState:
    """Cached eigensystem and oscillator metadata for one phase-slip sector."""

    sector: PhaseSlipSector
    evals: FloatArray
    evecs: ComplexArray

    osc: LocalHarmonicOscillator
    shift: float
    cutoff: int
    fluxonium: Any


class PhaseSlipModel:
    """Reduced single-mode model parameterized by phase-slip sector."""

    def __init__(self, params: TunnelingAssistedModelParams):
        """Initialize model with physical parameters.

        Parameters
        ----------
        params:
            Physical circuit parameters, external flux settings, and phase-slip
            amplitude ``E_S``.
        """
        self.params = params

    def inductive_fractions(self) -> tuple[float, float, float]:
        """Return inductive-energy sum and fractional weights.

        Returns
        -------
        tuple[float, float, float]
            ``(EL_sum, f_a, f_b)`` with ``f_a = EL_a / EL_sum`` and
            ``f_b = EL_b / EL_sum``.
        """
        EL_a = self.params.EL_a
        EL_b = self.params.EL_b
        EL_sum = EL_a + EL_b
        return EL_sum, EL_a / EL_sum, EL_b / EL_sum

    def phi_cm_diff(self) -> tuple[float, float]:
        """Return ``(phi_cm, phi_diff) = (phi_ext_a + phi_ext_b, phi_ext_a - phi_ext_b)``."""
        return (
            self.params.phi_ext_a + self.params.phi_ext_b,
            self.params.phi_ext_a - self.params.phi_ext_b,
        )

    def effective_flux(self, sector: PhaseSlipSector) -> float:
        """Effective external flux seen by the reduced 1D model for ``sector``."""
        _, f_a, f_b = self.inductive_fractions()
        phi_cm, phi_diff = self.phi_cm_diff()
        return (
            -2.0 * np.pi * f_a * sector.p + 0.5 * (f_b - f_a) * phi_cm - 0.5 * phi_diff
        )

    def energy_offset(self, sector: PhaseSlipSector) -> float:
        """Sector-dependent additive energy offset from the phase-slip model."""
        EL_sum, f_a, f_b = self.inductive_fractions()
        phi_cm, _ = self.phi_cm_diff()
        return 0.5 * EL_sum * f_a * f_b * (phi_cm + 2.0 * np.pi * sector.p) ** 2

    def coordinate_shift(self, sector: PhaseSlipSector) -> float:
        """Oscillator-center shift used in Fock-basis overlap calculations.

        For this model, the full effective flux is allocated onto the inductor.
        """
        return self.effective_flux(sector)

    def get_sector_key(self, sector: PhaseSlipSector) -> int:
        """Return dictionary / cache key for a sector."""
        return sector.p

    def get_scq_fluxonium_object(
        self, sector: PhaseSlipSector, cutoff: int = 120
    ) -> tuple[Any, float, LocalHarmonicOscillator]:
        """Construct scqubits Fluxonium for this sector.

        IMPORTANT:
        - scq uses flux in units of ``Phi0`` (i.e. ``2π = 1`` flux quantum).

        Parameters
        ----------
        sector:
            Phase-slip sector label.
        cutoff:
            Basis cutoff used to build the scqubits object.
        evals_count:
            Number of eigenpairs retained per sector.
        Returns
        -------
        tuple[Any, float, LocalHarmonicOscillator]
            ``(scq_fluxonium, coordinate_shift, local_oscillator)``.
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
            truncated_dim=cutoff,
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


class PhaseSlipSectorManager:
    """Lazy eigensystem cache over phase-slip sectors for a fixed model/cutoff."""

    def __init__(
        self,
        model: PhaseSlipModel,
        cutoff: int = 120,
        evals_count: int = 10,
    ):
        """Create manager with fixed diagonalization settings.

        Parameters
        ----------
        model:
            Phase-slip reduced model instance.
        cutoff:
            Basis cutoff for sector diagonalization.
        evals_count:
            Number of eigenpairs retained per sector.
        """
        self.model = model
        self.cutoff = cutoff
        self.evals_count = evals_count

        self._cache: dict[int, PhaseSlipSectorState] = {}

    def _key(self, sector: PhaseSlipSector) -> int:
        return sector.p

    def get_sector_state(self, sector: PhaseSlipSector) -> PhaseSlipSectorState:
        """Get (or build and cache) eigensystem data for ``sector``."""
        key = self._key(sector)

        if key in self._cache:
            return self._cache[key]

        scq_obj, shift, osc = self.model.get_scq_fluxonium_object(
            sector, cutoff=self.cutoff
        )

        evals_raw, evecs_raw = cast(
            tuple[NDArray[Any], NDArray[Any]],
            scq_obj.eigensys(  # pyright: ignore[reportUnknownMemberType]
                evals_count=self.evals_count
            ),
        )
        evals = np.asarray(evals_raw, dtype=np.float64)
        evecs = np.asarray(evecs_raw, dtype=np.complex128)

        state = PhaseSlipSectorState(
            sector=sector,
            evals=evals,
            evecs=evecs,
            shift=shift,
            cutoff=self.cutoff,
            osc=osc,
            fluxonium=scq_obj,
        )

        self._cache[key] = state
        return state
