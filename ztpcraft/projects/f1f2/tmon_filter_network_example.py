import numpy as np
import sympy as sm
import qutip as qt
import dynamiqs as dq
from scipy.constants import hbar
import copy
from pathlib import Path
from ztpcraft.toolbox.save import datetime_dir
import json

import dill

import jax.numpy as jnp

from ztpcraft.bosonic.input_output import (
    power_dissipator_coeff_frequency_to_drive_strength,
)

from pydantic import BaseModel, model_validator, field_validator

from typing import Callable, Any, Dict, Tuple, Literal, List, Union, Optional
from numpy.typing import NDArray

# -- Simulation-related classes --------------------------------------------------
"""LME simulation helper.

1. `DbdqsSimulation` – build full Hilbert-space Hamiltonian, transform to the
   dressed basis, truncate, time-evolve with *dynamiqs* and persist results.
2. `DbdqsAnalysis` – lightweight post-processing utilities.

Only the overall structure is included; places that depend on heavy physics
expressions are marked with `# TODO:` so users can paste the exact notebook
formulas.  Nothing here changes external public API later.
"""


###############################################################################
# Typed simulation configuration (validated via pydantic)                     #
###############################################################################


class PhysicalCfg(BaseModel):
    """Physical parameters for the system (large + truncated spaces)."""

    # Large Hilbert-space sizes (used for diagonalization)
    N_a_before_trunc: int = 10
    N_b_before_trunc: int = 15
    N_q_before_trunc: int = 10

    # Working sizes for real-time evolution
    N_a_trunc: int = 7
    N_b_trunc: int = 14
    N_q_trunc: int = 7

    # Bare mode frequencies (GHz)
    f_a_GHz: float
    f_b_GHz: float
    f_q_GHz: float

    # Josephson energy (GHz)
    EJ_GHz: float

    # Flux zpf factors (radian)
    phi_a_zpf: float
    phi_b_zpf: float
    phi_q_zpf: float

    # Loss / bath
    # c_op_a_prefactor, c_op_b_prefactor, c_op_q_prefactor appears as (at zero temperature)
    # D[sum_i c_op_<i>_prefactor * a_i], where a_i is the annihilation operator
    # for the i-th mode. Note that the dimensionality of this term is defined to be
    # compatible with a Hamiltonian in unit of GHz * 2 * pi.
    c_op_a_prefactor: float
    c_op_b_prefactor: float
    c_op_q_prefactor: float = 0.0
    n_bose: float = 0.0

    # Drive parameters (GHz)
    power_W: float
    eps_a_GHz: float
    eps_b_GHz: float
    eps_q_GHz: float
    use_power: bool = True
    # list of drive frequencies to be simulated
    drive_frequencies_GHz: List[float]

    # initial state
    initial_state: Dict[Tuple[int, ...], complex] = {(0, 0, 1): 1.0}

    # convert {"\"0,0,1\"": "1+0j"} → {(0,0,1): 1+0j}
    @field_validator("initial_state", mode="before")
    @classmethod
    def _parse_initial_state(cls, v):
        if isinstance(v, dict):
            parsed: Dict[Tuple[int, ...], complex] = {}
            for k, val in v.items():
                # key "0,0,1" -> tuple(0,0,1)
                if isinstance(k, str):
                    k_tuple = tuple(int(s) for s in k.split(","))  # (0,0,1)
                else:
                    k_tuple = tuple(k)
                # value "1+0j" -> 1+0j
                if isinstance(val, str):
                    val_cplx = complex(val.replace(" ", ""))
                else:
                    val_cplx = complex(val)
                parsed[k_tuple] = val_cplx
            return parsed
        return v

    # ------------------------------------------------------------------
    # Allow flexible specification of drive frequencies
    @field_validator("drive_frequencies_GHz", mode="before")
    @classmethod
    def _parse_drive_frequencies_GHz(cls, v):
        """Parse the *drive_frequencies_GHz* field.

        The user can either provide an explicit list/tuple of floats, or a
        dictionary describing a range from *start* to *stop* (both in GHz)
        together with either the number of points (*num*) or the step size
        (*step*).  Examples::

            [4.70, 4.75, 4.80]
            {"start": 4.70, "stop": 4.90, "num": 21}
            {"start": 4.70, "stop": 4.90, "step": 0.05}
        """
        # Plain list / tuple → ensure list[float]
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]

        # Dict form → build the sequence
        if isinstance(v, dict):
            if {"start", "stop", "num"} <= v.keys():
                start = float(v["start"])
                stop = float(v["stop"])
                num = int(v["num"])
                return np.linspace(start, stop, num).tolist()
            if {"start", "stop", "step"} <= v.keys():
                start = float(v["start"])
                stop = float(v["stop"])
                step = float(v["step"])
                # include the stop value within floating-point tolerance
                return np.arange(start, stop + step / 2, step).tolist()

        raise ValueError(
            "drive_frequencies_GHz must be a list of floats or a dict with "
            "keys {start, stop, num} or {start, stop, step}."
        )

    # Convenience
    @property
    def dim_before_trunc(self) -> int:  # noqa: D401
        return self.N_a_before_trunc * self.N_b_before_trunc * self.N_q_before_trunc

    @property
    def dim_trunc(self) -> int:  # noqa: D401
        return self.N_a_trunc * self.N_b_trunc * self.N_q_trunc


class SolutionCfg(BaseModel):
    """Time discretisation & pulse-shape parameters."""

    # processor
    processor: Literal["cpu", "gpu"] = "cpu"
    # single/double precision
    precision: Literal["single", "double"] = "single"

    # mode of simulation
    experiment_mode: Literal["Ramsey", "T1"] = "Ramsey"
    diagonal_level_to_measure: int = 5

    # time parameters
    t_ramp_nominal_ns: float = 1.0
    t_flat_nominal_ns: float = 120.0
    n_snapshots_per_period: int = 1
    include_ramp_pulse: bool = True
    lossy_system_during_ramp: bool = False

    # whether or not to displace the state for measurements
    displace_state_for_measurements: bool = False


class SolverCfg(BaseModel):
    method: str = "Tsit5"  # or "Expm"
    max_steps: int = 1000000000
    atol: float = 1e-6
    rtol: float = 1e-6


class OutputCfg(BaseModel):
    """Output directory configuration with auto timestamped sub-folder."""

    folder: Path = Path("./results")
    job: str | None = None

    @staticmethod
    def _make_job(parent: Path) -> str:
        parent.mkdir(parents=True, exist_ok=True)
        return datetime_dir(save_dir=str(parent), save_time=True)

    @model_validator(mode="after")
    def _auto_job(self):  # noqa: D401
        """Populate `job` with a timestamp sub-folder when it is missing."""
        if self.job is None:
            self.job = self._make_job(self.folder)
        return self


class SimulationCfg(BaseModel):
    physical_config: PhysicalCfg
    solution_config: SolutionCfg = SolutionCfg()
    solver_config: SolverCfg = SolverCfg()
    output_config: OutputCfg = OutputCfg()


###############################################################################
# Simulation                                                                  #
###############################################################################


class DriveInducedDecohSim:
    """End-to-end wrapper for the dressed-basis dynamical simulation."""

    def __init__(self, config: Union[str, Dict[str, Any]], autorun: bool = True):
        if isinstance(config, dict):
            # pydantic v2 uses `model_validate`
            self.cfg = SimulationCfg.model_validate(config)
        elif isinstance(config, str):
            self.cfg = SimulationCfg.model_validate_json(Path(config).read_text())
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        job_path = Path(self.cfg.output_config.job)
        if job_path.is_absolute():
            self.out_dir = job_path
        else:
            self.out_dir = (job_path).absolute()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(self.cfg.model_dump_json(indent=2))
        self._generate_truncation_projector_attr()
        self._generate_dqs_attr()
        self._generate_truncated_static_hamiltonian_attr()
        if autorun:
            self.static_circuit_setup()
            self.dynamic_circuit_setup()
            self.run_simulation()

    def static_circuit_setup(self):
        self._generate_evals_evecs_attr()
        self._generate_index_list_attr()
        self._generate_qubit_rho01_projector_attr()
        self._generate_qubit_diagonal_projector_attr()
        self._generate_truncation_projector_attr()

    def dynamic_circuit_setup(self):
        self._generate_solver_attr()
        self._generate_initial_state_attr()
        self._generate_c_ops_attr()
        self._generate_drive_strength_attr()

    # ---------------------------------------------------------------------
    # Attribute generators
    # ---------------------------------------------------------------------
    def _generate_solver_attr(self):
        self.solver = self._solver()

    def _generate_initial_state_attr(self):
        self.initial_state = self._generate_initial_state()

    def _generate_truncation_projector_attr(self):
        self.truncation_projector = self._generate_truncation_projector_from_dims(
            (
                self.cfg.physical_config.N_a_before_trunc,
                self.cfg.physical_config.N_b_before_trunc,
                self.cfg.physical_config.N_q_before_trunc,
            ),
            (
                self.cfg.physical_config.N_a_trunc,
                self.cfg.physical_config.N_b_trunc,
                self.cfg.physical_config.N_q_trunc,
            ),
        )

    def _generate_dqs_attr(self):
        self.destroy_ops_untruncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("destroy")
        )
        self.create_ops_untruncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("create")
        )
        self.number_ops_untruncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("number")
        )
        self.x_ops_untruncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("x")
        )
        self.y_ops_untruncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("y")
        )
        self.destroy_ops_truncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("destroy", truncated=True)
        )
        self.create_ops_truncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("create", truncated=True)
        )
        self.number_ops_truncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("number", truncated=True)
        )
        self.x_ops_truncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("x", truncated=True)
        )
        self.y_ops_truncated: Dict[Literal["a", "b", "q"], dq.QArray] = (
            self._generate_dqs_ops("y", truncated=True)
        )

    def _generate_truncated_static_hamiltonian_attr(self):
        self.static_hamiltonian_truncated = self._generate_static_hamiltonian(
            apply_truncation=True
        )

    def _generate_evals_evecs_attr(self):
        self.evals, self.evecs = self._diagonalize_hamiltonian()

    def _generate_index_list_attr(self):
        self.index_list = self._generate_index_list(self.evecs)

    def _generate_qubit_rho01_projector_attr(self):
        self.rho01_projector = self._generate_qubit_rho01_projector()

    def _generate_qubit_diagonal_projector_attr(self):
        self.diag_projector = {
            level: self._generate_qubit_diagonal_projector(qubit_level=level)
            for level in range(self.cfg.solution_config.diagonal_level_to_measure)
        }

    def _generate_c_ops_attr(self):
        self.c_ops = self._generate_c_ops()

    def _generate_drive_strength_attr(self):
        self.drive_strength = self._generate_drive_strength()

    # ---------------------------------------------------------------------
    # Operator helpers
    # ---------------------------------------------------------------------
    def _tensor_ops(
        self,
        mode_dims: Tuple[int, ...],
        package: Literal["qt", "dqs"],
        op_type: Literal["destroy", "create", "number", "x", "y"] = "destroy",
    ) -> Tuple[Union[qt.Qobj, dq.QArray], ...]:
        if package == "qt":
            destroy_op_func = qt.destroy
            eye_op_func = qt.qeye
            tensor_func = qt.tensor
        elif package == "dqs":
            destroy_op_func = dq.destroy
            eye_op_func = dq.eye
            tensor_func = dq.tensor
        else:
            raise ValueError(f"Invalid package: {package}")
        destroy_tensors: List[Union[qt.Qobj, dq.QArray]] = []

        # Pre-build single-mode operators for efficiency
        single_destroy = [destroy_op_func(dim) for dim in mode_dims]
        single_eye = [eye_op_func(dim) for dim in mode_dims]

        for target_idx in range(len(mode_dims)):
            ops = []
            for idx in range(len(mode_dims)):
                ops.append(
                    single_destroy[idx] if idx == target_idx else single_eye[idx]
                )
            ops = tuple(ops)
            destroy_tensors.append(tensor_func(*ops))
        if op_type == "destroy":
            return tuple(destroy_tensors)
        elif op_type == "create":
            return tuple([destroy_tensor.dag() for destroy_tensor in destroy_tensors])
        elif op_type == "number":
            return tuple(
                [
                    destroy_tensor.dag() @ destroy_tensor
                    for destroy_tensor in destroy_tensors
                ]
            )
        elif op_type == "x":
            return tuple(
                [
                    destroy_tensor.dag() + destroy_tensor
                    for destroy_tensor in destroy_tensors
                ]
            )
        elif op_type == "y":
            return tuple(
                [
                    (destroy_tensor.dag() - destroy_tensor) * 1j
                    for destroy_tensor in destroy_tensors
                ]
            )
        else:
            raise ValueError(f"Invalid op_type: {op_type}")

    def _cosm_op(self, op: Union[qt.Qobj, dq.QArray]) -> Union[qt.Qobj, dq.QArray]:
        return op.cosm()

    def _cos_nl_op(self, op: Union[qt.Qobj, dq.QArray]) -> Union[qt.Qobj, dq.QArray]:
        if isinstance(op, qt.Qobj):
            return op.cosm() - 1 * qt.qeye_like(op) + 0.5 * op @ op
        elif isinstance(op, dq.QArray):
            return op.cosm() - 1 * dq.eye_like(op) + 0.5 * op @ op

    def _generate_bare_labels(self, dims: tuple[int, ...]) -> NDArray[np.int64]:
        """Bare labels in the canonical order.
        E.g. for two qubits, returns [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        return np.array(list(np.ndindex(*dims)))

    def _generate_truncation_projector_from_dims(
        self, full_dims: tuple[int, ...], truncated_dims: tuple[int, ...]
    ) -> NDArray[np.float64]:
        """Projector onto the truncated subspace.

        Useful for projecting the Hamiltonian defined using a larger cutoff onto the
        subspace that we actually diagonalize and simulate.
        """
        bare_labels = self._generate_bare_labels(truncated_dims)

        return np.transpose(
            np.squeeze(
                [
                    qt.basis(list(full_dims), list(bare_label)).full()
                    for bare_label in bare_labels
                ]
            )
        )

    def _generate_dqs_ops(
        self,
        op_type: Literal["destroy", "create", "number", "x", "y"],
        truncated: bool = False,
    ) -> Dict[Literal["a", "b", "q"], dq.QArray]:
        if truncated:
            mode_dims = (
                self.cfg.physical_config.N_a_trunc,
                self.cfg.physical_config.N_b_trunc,
                self.cfg.physical_config.N_q_trunc,
            )
        else:
            mode_dims = (
                self.cfg.physical_config.N_a_before_trunc,
                self.cfg.physical_config.N_b_before_trunc,
                self.cfg.physical_config.N_q_before_trunc,
            )
        op_tuple = self._tensor_ops(
            mode_dims,
            package="dqs",
            op_type=op_type,
        )
        op_dict = {
            "a": op_tuple[0],
            "b": op_tuple[1],
            "q": op_tuple[2],
        }
        return op_dict

    # ---------------------------------------------------------------------
    # Hamiltonian construction (static + drive)
    # ---------------------------------------------------------------------
    def _generate_static_hamiltonian(
        self,
        apply_truncation: bool = False,
    ) -> dq.QArray:
        physical_params = self.cfg.physical_config
        n_a, n_b, n_q = (
            self.number_ops_untruncated["a"],
            self.number_ops_untruncated["b"],
            self.number_ops_untruncated["q"],
        )
        x_a, x_b, x_q = (
            self.x_ops_untruncated["a"],
            self.x_ops_untruncated["b"],
            self.x_ops_untruncated["q"],
        )
        phase_op = (
            physical_params.phi_a_zpf * x_a
            + physical_params.phi_b_zpf * x_b
            + physical_params.phi_q_zpf * x_q
        )
        cos_nl_phase_op = self._cos_nl_op(phase_op)
        phase_op = (
            physical_params.phi_a_zpf * x_a
            + physical_params.phi_b_zpf * x_b
            + physical_params.phi_q_zpf * x_q
        )
        cos_nl_phase_op = self._cos_nl_op(phase_op)
        static_hamiltonian = (
            2
            * np.pi
            * (
                physical_params.f_a_GHz * n_a
                + physical_params.f_b_GHz * n_b
                + physical_params.f_q_GHz * n_q
                - physical_params.EJ_GHz * cos_nl_phase_op
            )
        )
        if apply_truncation:
            static_hamiltonian = static_hamiltonian.to_numpy()
            static_hamiltonian = (
                self.truncation_projector.T
                @ static_hamiltonian
                @ self.truncation_projector
            )
            static_hamiltonian = dq.asqarray(
                static_hamiltonian,
                dims=(
                    self.cfg.physical_config.N_a_trunc,
                    self.cfg.physical_config.N_b_trunc,
                    self.cfg.physical_config.N_q_trunc,
                ),
            )
        return static_hamiltonian

    # ---------------------------------------------------------------------
    # Diagonzation and generation of projectors
    # ---------------------------------------------------------------------
    def _diagonalize_hamiltonian(self) -> Tuple[NDArray[np.float64], List[dq.QArray]]:
        hamiltonian_ndarray: NDArray[np.float64] = (
            self.static_hamiltonian_truncated.to_numpy()
        )
        evals, evecs = np.linalg.eigh(hamiltonian_ndarray)
        # standardize the eigenvectors to have positive real part of the largest element
        evecs_new_dq = []
        for evec in evecs.T:
            evec_full = copy.deepcopy(evec)
            if evec[np.argmax(np.abs(evec))] < 0:
                evec_full *= -1
            evecs_new_dq.append(
                dq.asqarray(
                    np.array([evec_full]),
                    dims=(
                        self.cfg.physical_config.N_a_trunc,
                        self.cfg.physical_config.N_b_trunc,
                        self.cfg.physical_config.N_q_trunc,
                    ),
                )
            )
        return evals, evecs_new_dq

    def _generate_index_list(self, evecs: List[dq.QArray]) -> NDArray[np.int64]:
        index_list = np.zeros((self.cfg.physical_config.dim_trunc, 4))
        for i in range(self.cfg.physical_config.dim_trunc):
            state = evecs[i]
            index_list[i, :] = [
                i,
                int(
                    np.real(round(dq.expect(self.number_ops_truncated["a"], state), 0))
                ),
                int(
                    np.real(round(dq.expect(self.number_ops_truncated["b"], state), 0))
                ),
                int(
                    np.real(round(dq.expect(self.number_ops_truncated["q"], state), 0))
                ),
            ]
        return index_list

    def _get_index(
        self, a_state: int, b_state: int, q_state: int, index_list: NDArray[np.int64]
    ) -> int:
        index = None
        diff = 1e4
        for i in range(self.cfg.physical_config.dim_trunc):
            if (
                index_list[i, 1] == a_state
                and index_list[i, 2] == b_state
                and np.abs(index_list[i, 3] - q_state) < diff
            ):
                index = i
                diff = np.abs(index_list[i, 3] - q_state)
        return index

    def _generate_qubit_rho01_projector(self) -> dq.QArray:
        off_diag_proj: dq.QArray = 0 * dq.eye_like(self.evecs[0])
        for i in range(self.cfg.physical_config.N_a_trunc):
            for j in range(self.cfg.physical_config.N_b_trunc):
                state_ij0 = self.evecs[self._get_index(i, j, 0, self.index_list)]
                state_ij1 = self.evecs[self._get_index(i, j, 1, self.index_list)]
                off_diag_proj += dq.dag(state_ij0) @ state_ij1
        return off_diag_proj

    def _generate_qubit_diagonal_projector(self, qubit_level: int) -> dq.QArray:
        diag_proj: dq.QArray = 0 * dq.eye_like(self.evecs[0])
        for i in range(self.cfg.physical_config.N_a_trunc):
            for j in range(self.cfg.physical_config.N_b_trunc):
                state_ijk = self.evecs[
                    self._get_index(i, j, qubit_level, self.index_list)
                ]
                diag_proj += dq.dag(state_ijk) @ state_ijk
        return diag_proj

    # ---------------------------------------------------------------------
    # Solver wrapper
    # ---------------------------------------------------------------------
    def _solver(self):
        s = self.cfg.solver_config
        if s.method.lower() == "tsit5":
            return dq.integrators.Tsit5(max_steps=s.max_steps, atol=s.atol, rtol=s.rtol)
        return dq.integrators.Expm()

    def compute_chi(self) -> Dict[Literal["ab", "aq", "bq"], float]:
        """
        Return chi in unit of GHz
        """
        state_idx_000 = self._get_index(0, 0, 0, self.index_list)
        state_idx_001 = self._get_index(0, 0, 1, self.index_list)
        state_idx_010 = self._get_index(0, 1, 0, self.index_list)
        state_idx_011 = self._get_index(0, 1, 1, self.index_list)
        state_idx_100 = self._get_index(1, 0, 0, self.index_list)
        state_idx_101 = self._get_index(1, 0, 1, self.index_list)
        state_idx_110 = self._get_index(1, 1, 0, self.index_list)
        chi_ab = (
            (
                self.evals[state_idx_110]
                - self.evals[state_idx_010]
                - self.evals[state_idx_100]
                + self.evals[state_idx_000]
            )
            / 2
            / np.pi
        )
        chi_aq = (
            (
                self.evals[state_idx_101]
                - self.evals[state_idx_001]
                - self.evals[state_idx_100]
                + self.evals[state_idx_000]
            )
            / 2
            / np.pi
        )
        chi_bq = (
            (
                self.evals[state_idx_011]
                - self.evals[state_idx_001]
                - self.evals[state_idx_010]
                + self.evals[state_idx_000]
            )
            / 2
            / np.pi
        )
        return {"ab": chi_ab, "aq": chi_aq, "bq": chi_bq}

    def _generate_initial_state(self) -> dq.QArray:
        init_state_dict = self.cfg.physical_config.initial_state
        init_state = np.zeros(self.cfg.physical_config.dim_trunc)
        for state_label, amplitude in init_state_dict.items():
            init_state += (
                amplitude * self.evecs[self._get_index(*state_label, self.index_list)]
            )
        init_state = init_state / init_state.norm()
        return init_state

    def _generate_c_ops(self) -> List[dq.QArray]:
        n_bose = self.cfg.physical_config.n_bose
        c_prefactor_a = self.cfg.physical_config.c_op_a_prefactor
        c_prefactor_b = self.cfg.physical_config.c_op_b_prefactor
        c_prefactor_q = self.cfg.physical_config.c_op_q_prefactor
        c_ops = [
            np.sqrt(1 + n_bose)
            * (
                c_prefactor_a * self.number_ops_truncated["a"]
                + c_prefactor_b * self.number_ops_truncated["b"]
                + c_prefactor_q * self.number_ops_truncated["q"]
            ),
            np.sqrt(n_bose)
            * (
                c_prefactor_a * self.number_ops_truncated["a"]
                + c_prefactor_b * self.number_ops_truncated["b"]
                + c_prefactor_q * self.number_ops_truncated["q"]
            ),
        ]
        return c_ops

    def _generate_drive_strength(
        self,
    ) -> Dict[Literal["a", "b", "q"], NDArray[np.float64]]:
        physical_config = self.cfg.physical_config
        drive_strength_dict: Dict[Literal["a", "b", "q"], NDArray[np.float64]] = {}
        for mode in ["a", "b", "q"]:
            if physical_config.use_power:
                drive_strength = np.zeros(len(physical_config.drive_frequencies_GHz))
                for idx, f_GHz in enumerate(physical_config.drive_frequencies_GHz):
                    if mode == "a":
                        c_op_prefactor = physical_config.c_op_a_prefactor
                    elif mode == "b":
                        c_op_prefactor = physical_config.c_op_b_prefactor
                    elif mode == "q":
                        c_op_prefactor = physical_config.c_op_q_prefactor
                    else:
                        raise ValueError(f"Invalid mode: {mode}")

                    drive_strength[idx] = (
                        power_dissipator_coeff_frequency_to_drive_strength(
                            input_power=physical_config.power_W,
                            dissipator_coeff=c_op_prefactor,
                            omega_d=f_GHz * 2 * np.pi,
                        )
                    )
                drive_strength_dict[mode] = drive_strength
            else:
                if mode == "a":
                    drive_strength = physical_config.eps_a_GHz
                elif mode == "b":
                    drive_strength = physical_config.eps_b_GHz
                elif mode == "q":
                    drive_strength = physical_config.eps_q_GHz
                drive_strength_dict[mode] = (
                    np.ones(len(physical_config.drive_frequencies_GHz))
                    * drive_strength
                    * 2
                    * np.pi
                )
        return drive_strength_dict

    def run_simulation(self) -> Path:
        dq.set_device(self.cfg.solution_config.processor)
        dq.set_precision(self.cfg.solution_config.precision)

        initial_state = self.initial_state.dag()
        static_hamiltonian = self.static_hamiltonian_truncated
        physical_config = self.cfg.physical_config
        solution_config = self.cfg.solution_config
        drive_frequencies_GHz = physical_config.drive_frequencies_GHz
        simulation_outcome_dict_list = []

        for freq_idx, f_GHz in enumerate(drive_frequencies_GHz):
            print(f"frequency sweep {freq_idx+1}/{len(drive_frequencies_GHz)}")
            omega_d = f_GHz * 2 * np.pi
            # round to nearest 1/f_d time for the ramp and the flat part
            t_ramp_rounded = np.round(solution_config.t_ramp_nominal_ns * f_GHz) / f_GHz
            t_flat_rounded = np.round(solution_config.t_flat_nominal_ns * f_GHz) / f_GHz
            t_period = 1 / f_GHz
            t_sample_step = t_period / solution_config.n_snapshots_per_period
            n_snapshots_ramp = int(t_ramp_rounded / t_sample_step)
            n_snapshots_flat = int(t_flat_rounded / t_sample_step)

            if solution_config.include_ramp_pulse:
                print("Evolving for ramping up")
                tlist_ramp = np.linspace(0, t_ramp_rounded, n_snapshots_ramp + 1)

                def pulse_ramp_up(
                    t, args: Dict[str, float], mode: Literal["a", "b", "q"]
                ) -> float:
                    t_ramp = args["t_ramp"]
                    return (
                        self.drive_strength[mode][freq_idx]
                        * jnp.cos(omega_d * t)
                        * (t / t_ramp)
                    )

                H_d_ramp_up = static_hamiltonian
                for mode in ["a", "b", "q"]:
                    if self.drive_strength[mode][freq_idx] != 0:
                        H_d_ramp_up += dq.modulated(
                            # capture *mode* to avoid late-binding pitfalls in Python loops
                            lambda t, m=mode: pulse_ramp_up(
                                t, args={"t_ramp": t_ramp_rounded}, mode=m
                            ),
                            self.y_ops_truncated[mode],
                        )

                print("Computing ramp propagators")
                if not solution_config.lossy_system_during_ramp:
                    ramp_up_output = dq.sesolve(
                        H_d_ramp_up,
                        initial_state,
                        tlist_ramp,
                        method=self.solver,
                        exp_ops=[
                            self.number_ops_truncated["a"],
                            self.number_ops_truncated["b"],
                            self.number_ops_truncated["q"],
                        ],
                        options=dq.Options(save_states=True),
                    )
                else:
                    # evolve the state for the ramp up
                    ramp_up_output = dq.mesolve(
                        H_d_ramp_up,
                        self.c_ops,
                        initial_state,
                        tlist_ramp,
                        method=self.solver,
                        exp_ops=[
                            self.number_ops_truncated["a"],
                            self.number_ops_truncated["b"],
                            self.number_ops_truncated["q"],
                        ],
                        options=dq.Options(save_states=True),
                    )

            print("Evolving for flat part")
            tlist_flat = np.linspace(0, t_flat_rounded, n_snapshots_flat + 1)

            def pulse_flat(
                t, args: Dict[str, float], mode: Literal["a", "b", "q"]
            ) -> float:
                return self.drive_strength[mode][freq_idx] * jnp.cos(omega_d * t)

            H_d_flat = static_hamiltonian
            for mode in ["a", "b", "q"]:
                if self.drive_strength[mode][freq_idx] != 0:
                    H_d_flat += dq.modulated(
                        # capture *mode* to avoid late-binding pitfalls in Python loops
                        lambda t, m=mode: pulse_flat(
                            t, args={"t_flat": t_flat_rounded}, mode=m
                        ),
                        self.y_ops_truncated[mode],
                    )

            if solution_config.include_ramp_pulse:
                initial_state_for_flat_pulse = ramp_up_output.states[-1]
            else:
                initial_state_for_flat_pulse = initial_state.dag()
            flat_output = dq.mesolve(
                H_d_flat,
                self.c_ops,
                initial_state_for_flat_pulse,
                tlist_flat,
                method=self.solver,
                exp_ops=[
                    self.number_ops_truncated["a"],
                    self.number_ops_truncated["b"],
                    self.number_ops_truncated["q"],
                ],
                options=dq.Options(save_states=True),
            )
            measurement_ops: Dict[str, dq.QArray] = {}
            if solution_config.experiment_mode == "Ramsey":
                measurement_ops["01"] = self.rho01_projector
            elif solution_config.experiment_mode == "T1":
                for level_idx, projector in self.diag_projector.items():
                    measurement_ops[f"{level_idx}{level_idx}"] = projector
            else:
                raise ValueError(
                    f"Invalid experiment mode: {solution_config.experiment_mode}"
                )

            measurement_outcomes: Dict[str, NDArray[np.complex128]] = {}
            for projector_label, projector in measurement_ops.items():
                measurement_outcomes[projector_label] = np.zeros(
                    len(tlist_flat), dtype=np.complex128
                )
                for time_idx, time in enumerate(tlist_flat):
                    if solution_config.displace_state_for_measurements:
                        raise NotImplementedError(
                            "Displacement of state for measurements is not implemented"
                        )
                    else:
                        measurement_outcomes[projector_label][time_idx] = dq.expect(
                            projector, flat_output.states[time_idx]
                        )

            simulation_outcome_dict_list.append(
                {
                    # parameters
                    "drive_frequency_GHz": f_GHz,
                    "tlist_flat": tlist_flat,
                    "flat_a_expect": flat_output.expects[0],
                    "flat_b_expect": flat_output.expects[1],
                    "flat_q_expect": flat_output.expects[2],
                    "measurement_outcomes": measurement_outcomes,
                }
            )
            if solution_config.include_ramp_pulse:
                simulation_outcome_dict_list[freq_idx].update(
                    {
                        "tlist_ramp": tlist_ramp,
                        "ramp_up_a_expect": ramp_up_output.expects[0],
                        "ramp_up_b_expect": ramp_up_output.expects[1],
                        "ramp_up_q_expect": ramp_up_output.expects[2],
                    }
                )
            print(
                f"Saved data for {f_GHz:.4f} GHz ({freq_idx+1}/{len(drive_frequencies_GHz)})"
            )
        simulation_data_dict: Dict[str, Any] = {
            "outcome": simulation_outcome_dict_list,
            "config": self.cfg,
        }

        with open(self.out_dir / "simulation_data.pkl", "wb") as f:
            dill.dump(simulation_data_dict, f)

        self.simulation_data_dict = simulation_data_dict

        return self.out_dir

    # ------------------------------------------------------------------
    # Convenience helpers (public API)
    # ------------------------------------------------------------------
    @classmethod
    def generate_config_template(
        cls,
        save_path: str | Path | None = None,
        output_type: Literal["json", "dict"] = "json",
        *,
        indent: int = 4,
    ) -> Union[str, Dict[str, Any]]:
        """Return a JSON string with a *minimal* configuration template.

        Parameters
        ----------
        save_path
            When given, the JSON template is also written to the specified
            file path (parent directories are created automatically).
        indent
            Number of spaces for `json.dumps` pretty-printing.

        Returns
        -------
        str
            The JSON text that was generated (and possibly written to disk).
        """

        # Build a partially-filled config dict illustrating the required keys.
        template: Dict[str, Any] = {
            "physical_config": {
                # Hilbert-space sizes
                "N_a_before_trunc": 7,
                "N_b_before_trunc": 7,
                "N_q_before_trunc": 7,
                "N_a_trunc": 5,
                "N_b_trunc": 5,
                "N_q_trunc": 5,
                # Bare frequencies (GHz) – *** EDIT THESE ***
                "f_a_GHz": 5.000,
                "f_b_GHz": 6.200,
                "f_q_GHz": 4.800,
                # Josephson energy & ZPF factors
                "EJ_GHz": 20.0,
                "phi_a_zpf": 0.015,
                "phi_b_zpf": 0.017,
                "phi_q_zpf": 0.220,
                # Dissipation prefactors (GHz·2π)
                "c_op_a_prefactor": 0.0004,
                "c_op_b_prefactor": 0.0001,
                "c_op_q_prefactor": 0.0,
                "n_bose": 0.0,
                # Drive parameters
                "use_power": True,
                "power_W": 1e-9,
                "eps_a_GHz": 0.0,
                "eps_b_GHz": 0.0,
                "eps_q_GHz": 0.0,
                # Either supply an explicit list or a dict describing a linspace/arange
                # "drive_frequencies_GHz": [4.70, 4.80, 4.90],
                # "drive_frequencies_GHz": {"start": 4.70, "stop": 4.90, "step": 0.05},
                "drive_frequencies_GHz": {"start": 4.70, "stop": 4.90, "num": 21},
                # Initial state (ket amplitudes)
                "initial_state": {"0, 0, 1": "1.0 + 0j"},
            },
            "solution_config": {
                "processor": "cpu",
                "precision": "single",
                "experiment_mode": "Ramsey",
                "diagonal_level_to_measure": 5,
                "t_ramp_nominal_ns": 1.0,
                "t_flat_nominal_ns": 120.0,
                "n_snapshots_per_period": 1,
                "include_ramp_pulse": True,
                "lossy_system_during_ramp": False,
                "displace_state_for_measurements": False,
            },
            "solver_config": {
                "method": "Tsit5",
                "max_steps": 1_000_000_000,
                "atol": 1e-6,
                "rtol": 1e-6,
            },
            "output_config": {
                "folder": "./results",
                "job": None,
            },
        }

        json_text = json.dumps(template, indent=indent)

        if save_path is not None:
            save_path = Path(save_path).expanduser().absolute()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json_text)
        if output_type == "json":
            return json_text
        elif output_type == "dict":
            return template
        else:
            raise ValueError(f"Invalid output type: {output_type}")


###############################################################################
# Analysis                                                                    #
###############################################################################


class DriveInducedDecohPostProc:
    """Minimal post-processing helper (load files, quick plots, etc.)."""

    def __init__(self, result_dir: str | Path):
        self.dir = Path(result_dir).expanduser().absolute()
        self.files = sorted(self.dir.glob("simulation_data.pkl"))
        if not self.files:
            raise FileNotFoundError(f"No simulation_data.pkl in {self.dir}")
        self.simulation_data_dict = self._load(0)
        self.config = self.simulation_data_dict["config"]
        self.outcome_list = self.simulation_data_dict["outcome"]

    def _load(self, idx: int):
        with open(self.files[idx], "rb") as f:
            return dill.load(f)

    def max_amplitudes(self) -> Dict[float, float]:
        amps: Dict[float, float] = {}
        for f in self.files:
            data = np.load(f)
            amp = np.max(np.abs(data["expect"]))
            freq = float(f.stem.split("_")[1])
            amps[freq] = amp
        return amps

    def quick_plot(self, idx: int = 0, *, show: bool = True):
        import matplotlib.pyplot as plt

        d = self._load(idx)
        t_us = d["t"] * 1e6
        y = np.abs(d["expect"])
        plt.plot(t_us, y)
        plt.xlabel("Time (µs)")
        plt.ylabel("|a(t)|")
        plt.title(self.files[idx].stem)
        if show:
            plt.show()
        else:
            plt.savefig(self.dir / f"amp_{idx}.png", dpi=300)

    # ------------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------------
    def collect_results(self) -> Dict[str, Any]:
        """Return a nested dict organised as::

            {
                "flat_a_expect": {freq: np.ndarray, ...},
                "flat_b_expect": {freq: np.ndarray, ...},
                "flat_q_expect": {freq: np.ndarray, ...},
                "measurement_outcomes": {
                    "01": {freq: np.ndarray, ...},
                    "11": {freq: np.ndarray, ...},
                    ...
                },
            }

        where *freq* is the drive frequency in GHz and the arrays are the
        time-series saved by the simulation.
        """

        outcome_list = self.outcome_list

        # Initialise the container with the expected top-level keys.
        result: Dict[str, Any] = {
            "tlist_flat": {},
            "flat_a_expect": {},
            "flat_b_expect": {},
            "flat_q_expect": {},
            "measurement_outcomes": {},
        }

        for rec in outcome_list:
            # Skip ramp records – they lack the *flat_* keys.
            if "flat_a_expect" not in rec:
                continue

            freq = float(rec["drive_frequency_GHz"])

            # Simple quantities ------------------------------------------------
            result["flat_a_expect"][freq] = rec["flat_a_expect"]
            result["flat_b_expect"][freq] = rec["flat_b_expect"]
            result["flat_q_expect"][freq] = rec["flat_q_expect"]

            # Nested measurement outcomes -------------------------------------
            for label, ts in rec["measurement_outcomes"].items():
                if label not in result["measurement_outcomes"]:
                    result["measurement_outcomes"][label] = {}
                result["measurement_outcomes"][label][freq] = ts

        return result


# -- Helper functions ------------------------------------------------------------


def three_resonator_capacitive_coupling_secular_equation(
    f_01: float,
    f_02: float,
    f_0q: float,
    kappa: float,
    g_12: float,
    g_t1: float,
    g_t2: float,
) -> sm.Expr:
    """
    construct the secular equation to be solved for the eigenmode frequencies of a three-mode system
    where each mode is coupled to the other two modes via charge coupling. The frequencies and kappa
    can be in radian units or frequency units, as long as they are consistent. For example, if
    f_01 and f_02 are in radian units, then kappa is the inverse of the photon lifetime. The lossy port
    is defined as the port that is coupled to the filter mode 2.

    Parameters
    ----------
    f_01 : float
        frequency of the first filter mode
    f_02 : float
        frequency of the second filter mode
    f_0q : float
        frequency of the qubit mode
    kappa : float
        2x the damping rate of the second filter mode (writing the 2nd bare mode frequency as
        f_02 + i*kappa/2)
    g_12 : float
        charge coupling coefficient between the two filters; defined as g_12 = C_12/sqrt(C_1*C_2), where
        C_12 is the coupling capacitance, and C_1 and C_2 are the self-capacitances of the two filters
    g_t1 : float
        charge coupling coefficient between the first filter and the qubit
    g_t2 : float
        charge coupling coefficient between the second filter and the qubit

    Returns
    -------
    polynomial : sympy expression
        the polynomial to be solved for the eigenmode frequencies
    """
    x = sm.symbols("x")
    C_ratio = np.sqrt(g_t1 * g_t2 * g_12)
    return (
        (-(x**2) + f_01**2) * (-(x**2) + f_02**2 + 1j * kappa * x) * (-(x**2) + f_0q**2)
        + 2 * x**6 * C_ratio
        - (-(x**2) + f_01**2) * x**4 * g_t2
        - (-(x**2) + f_02**2 + 1j * kappa * x) * x**4 * g_t1
        - (-(x**2) + f_0q**2) * x**4 * g_12
    )


def mode_freqs_from_secular_equation(
    *args: tuple, secular_equation: Callable
) -> np.ndarray:
    """
    obtain the complex frequencies of the system given the parameters

    Parameters
    ----------
    *args : tuple
        tuple of parameters to be used in the secular equation
    secular_equation : Callable
        the secular equation to be solved for the mode frequencies

    Returns
    -------
    sorted_roots : array
        array of complex roots of the polynomial sorted in ascending order of real
        parts
    """
    # pass through the arguments to the secular equation
    poly = secular_equation(*args)
    root_list = list(sm.roots(poly).keys())
    # obtain the real part of the roots, and get the order of the roots in ascending order
    real_parts = [complex(root).real for root in root_list]
    sorted_indices = np.argsort(real_parts)
    sorted_roots = np.array(root_list)[sorted_indices]
    # convert sympy complex to numpy complex
    sorted_roots = np.array([complex(root) for root in sorted_roots])[3:]
    return sorted_roots


def three_resonator_dressed_basis_langevin_matrix(
    omega_1: float, omega_2: float, omega_3: float, c1: float, c2: float, c3: float
) -> np.ndarray:
    """
    construct the matrix for the Langevin equation for the three-mode system in the dressed basis.
    The matrix M is defined as
    d a_i/dt = sum_j M_ij a_j
    where a_i is the annihilation operator for the i-th mode.
    The setting is that the dissipator takes the form (at zero temperature) D[sum_i c_i * a_i]

    Parameters
    ----------
    omega_1 : float
        frequency of the first mode in angular units
    omega_2 : float
        frequency of the second mode in angular units
    omega_3 : float
        frequency of the third mode in angular units
    c1 : float
        the coefficient of the first mode in the dissipator
    c2 : float
        the coefficient of the second mode in the dissipator
    c3 : float
        the coefficient of the third mode in the dissipator

    Returns
    -------
    M : np.ndarray
        the matrix for the Langevin equation
    """
    M = np.array(
        [
            [
                -1j * omega_1 - 0.5 * c1 * np.conj(c1),
                -0.5 * c1 * np.conj(c2),
                -0.5 * c1 * np.conj(c3),
            ],
            [
                -0.5 * c2 * np.conj(c1),
                -1j * omega_2 - 0.5 * c2 * np.conj(c2),
                -0.5 * c2 * np.conj(c3),
            ],
            [
                -0.5 * c3 * np.conj(c1),
                -0.5 * c3 * np.conj(c2),
                -1j * omega_3 - 0.5 * c3 * np.conj(c3),
            ],
        ],
        dtype=complex,
    )
    return M


def mode_freqs_from_LME(
    omega_1: float, omega_2: float, omega_3: float, c1: float, c2: float, c3: float
) -> tuple:
    """
    obtain the complex frequencies of the system given the parameters

    Parameters
    ----------
    omega_1 : float
        frequency of the first mode in angular units
    omega_2 : float
        frequency of the second mode in angular units
    omega_3 : float
        frequency of the third mode in angular units
    c1 : float
        the coefficient of the first mode in the dissipator
    c2 : float
        the coefficient of the second mode in the dissipator
    c3 : float
        the coefficient of the third mode in the dissipator

    Returns
    -------
    sorted_eigvals : tuple
        tuple of the complex frequencies of the three modes in ascending order of imaginary part
    """
    M = three_resonator_dressed_basis_langevin_matrix(
        omega_1, omega_2, omega_3, c1, c2, c3
    )
    eigvals = np.linalg.eigvals(-M)
    # the imaginary part is the frequency, real part is 1/2 kappa
    # sort them in ascending order of imaginary part
    sorted_indices = np.argsort(eigvals.imag)
    sorted_eigvals = eigvals[sorted_indices]
    return (
        sorted_eigvals[0].imag + 1j * sorted_eigvals[0].real,
        sorted_eigvals[1].imag + 1j * sorted_eigvals[1].real,
        sorted_eigvals[2].imag + 1j * sorted_eigvals[2].real,
    )


def squared_energy_differences(
    x: tuple, func_for_mode_freqs: Callable, target_complex_freqs: tuple
) -> float:
    """
    calculate the squared energy differences between the obtained complex frequencies
    and the target complex frequencies. Currently only implemented for three-mode systems.
    The objective function is defined as the sum of relative errors in the real and imaginary parts
    of the complex frequencies, excluding the imaginary part of the qubit mode.

    Parameters
    ----------
    x : tuple
        tuple of parameters to be used for determining frequencies
    func_for_mode_freqs : Callable
        function for obtaining the complex frequencies of the system
    target_complex_freqs : tuple
        tuple of target complex frequencies

    Returns
    -------
    squared energy differences : float
        sum of the squared energy differences between the obtained complex frequencies
        and the target complex frequencies
    """
    # pass through the arguments to the secular equation
    computed_complex_freqs = func_for_mode_freqs(*x)
    # compute the squared energy differences
    f1, f2, f3 = computed_complex_freqs
    target_f1, target_f2, target_f3 = target_complex_freqs
    return (
        ((f1.real - target_f1.real) / target_f1.real) ** 2
        + ((f2.real - target_f2.real) / target_f2.real) ** 2
        + ((f3.real - target_f3.real) / target_f3.real) ** 2
        + ((f1.imag - target_f1.imag) / target_f1.imag) ** 2
        + ((f2.imag - target_f2.imag) / target_f2.imag) ** 2
    )


def bare_basis_eom(
    y: np.ndarray,
    t: float,
    omega_tuple: tuple,
    g_tuple: tuple,
    kappa: float,
    F: float,
    omega_d: float,
) -> np.ndarray:
    """
    define the equations of motion for the bare basis of a three-mode system with capacitive coupling.
    The equations of motion are defined in the form of a second order differential equation.

    Parameters
    ----------
    y : np.ndarray
        array of the variables to be solved for, in the order of x1, x2, x3, dx1dt, dx2dt, dx3dt
    t : float
        time
    omega_tuple : tuple
        tuple of the frequencies of the three modes in angular units
    g_tuple : tuple
        tuple of the coupling coefficients between the three modes, in the order of g_12_1, g_12_2, g_t1_t, g_t1_1, g_t2_t, g_t2_2
        gij_k is computed by evaluating Cij/Ck. Ck is the self-capacitance of the k-th mode, and Cij is the coupling capacitance
        between the i-th and j-th modes.
    kappa : float
        Inverse of photon lifetime of the second filter mode
    F : float
        drive strength in angular units
    omega_d : float
        drive frequency in angular units
    """
    x1, x2, x3, dx1dt, dx2dt, dx3dt = y
    omega_01, omega_02, omega_0q = omega_tuple
    g12_1, g12_2, gt1_t, gt1_1, gt2_t, gt2_2 = g_tuple
    d2xdt2_before_inv = np.array(
        [
            -(omega_01**2) * x1,
            -(omega_02**2) * x2 - kappa * dx2dt + F * np.sin(omega_d * t),
            -(omega_0q**2) * x3,
        ]
    )
    M_matrix = np.array(
        [
            [1, -g12_1, -gt1_1],
            [-g12_2, 1, -gt2_2],
            [-gt1_t, -gt2_t, 1],
        ]
    )
    d2xdt2 = np.linalg.inv(M_matrix) @ d2xdt2_before_inv
    dydt = np.array([dx1dt, dx2dt, dx3dt, d2xdt2[0], d2xdt2[1], d2xdt2[2]])
    return dydt


def dressed_basis_eom(
    y: np.ndarray,
    t: float,
    omega_tuple: tuple,
    c_tuple: tuple,
    eps: float,
    omega_d: float,
) -> np.ndarray:
    """
    define the equations of motion for the dressed basis of a three-mode system with capacitive coupling.
    The equations of motion are defined in the form of a second order differential equation.

    Parameters
    ----------
    y : np.ndarray
        array of the variables to be solved for, in the order of a1, a2, aq
    t : float
        time
    omega_tuple : tuple
        tuple of the frequencies of the three modes in angular units
    c_tuple : tuple
        tuple of the coefficients of the three modes in the dissipator
    eps : float
        drive strength in angular units
    omega_d : float
        drive frequency in angular units

    Returns
    -------
    dydt : np.ndarray
        array of the time derivatives of the variables
    """
    a1, a2, aq = y
    omega_01, omega_02, omega_0q = omega_tuple
    c1, c2, c3 = c_tuple
    dydt = np.array(
        [
            [
                -1j * (omega_01) - 0.5 * c1 * np.conj(c1),
                -0.5 * c1 * np.conj(c2),
                -0.5 * c1 * np.conj(c3),
            ],
            [
                -0.5 * c2 * np.conj(c1),
                -1j * (omega_02) - 0.5 * c2 * np.conj(c2),
                -0.5 * c2 * np.conj(c3),
            ],
            [
                -0.5 * c3 * np.conj(c1),
                -0.5 * c3 * np.conj(c2),
                -1j * (omega_0q) - 0.5 * c3 * np.conj(c3),
            ],
        ],
        dtype=complex,
    ) @ np.array([a1, a2, aq]) + np.array([c1, c2, c3]) * eps * np.cos(1j * omega_d * t)
    return dydt


def _get_drive_amplitude(
    P: float,
    omega_d: float,
    omega1_opt: float,
    omega2_opt: float,
    omega3_opt: float,
    c1_opt: float,
    c2_opt: float,
    c3_opt: float,
) -> tuple:
    """
    Has not been verified yet.
    Extract the drive amplitude for the three-mode system.
    """

    def heisenberg_LHS(x):
        omega_d = x[0]
        omega_1 = x[1]
        omega_2 = x[2]
        omega_3 = x[3]
        c1 = x[4]
        c2 = x[5]
        c3 = x[6]
        M = np.array(
            [
                [
                    -1j * (omega_1 - omega_d) - 0.5 * c1 * np.conj(c1),
                    -0.5 * c1 * np.conj(c2),
                    -0.5 * c1 * np.conj(c3),
                ],
                [
                    -0.5 * c2 * np.conj(c1),
                    -1j * (omega_2 - omega_d) - 0.5 * c2 * np.conj(c2),
                    -0.5 * c2 * np.conj(c3),
                ],
                [
                    -0.5 * c3 * np.conj(c1),
                    -0.5 * c3 * np.conj(c2),
                    -1j * (omega_3 - omega_d) - 0.5 * c3 * np.conj(c3),
                ],
            ],
            dtype=complex,
        )
        return M

    A_mat = -heisenberg_LHS(
        [omega_d, omega1_opt, omega2_opt, omega3_opt, c1_opt, c2_opt, c3_opt]
    )
    inv_A_mat = np.linalg.inv(A_mat)
    c_vec = np.array([c1_opt, c2_opt, c3_opt])

    B_vec = (1 - c_vec.conj().T @ inv_A_mat @ c_vec) * (-1 / c_vec[0]) * A_mat[0, :]

    g_a = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[0]
        / 2
        / np.pi
    )
    g_b = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[1]
        / 2
        / np.pi
    )
    g_q = (
        np.sqrt(
            P
            / (hbar * omega_d)
            / (
                (np.abs(B_vec[0] * inv_A_mat[0, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[1] * inv_A_mat[1, :] @ c_vec / 2)) ** 2
                + (np.abs(B_vec[2] * inv_A_mat[2, :] @ c_vec / 2)) ** 2
            )
        )
        / 1e9
        * c_vec[2]
        / 2
        / np.pi
    )
    return g_a, g_b, g_q
