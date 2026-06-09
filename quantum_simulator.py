"""
================================================================================
  FOUR-QUBIT QUANTUM COMPUTING SIMULATOR
  A fully annotated, educational implementation in Python / NumPy
================================================================================

BACKGROUND — WHY THIS FILE EXISTS
-----------------------------------
Quantum computers exploit two phenomena that have no classical analogue:
  • Superposition  — a qubit can be BOTH |0⟩ and |1⟩ simultaneously until
                     measured.
  • Entanglement   — two or more qubits can share a joint quantum state such
                     that measuring one instantly determines the other,
                     regardless of distance.

This simulator models both phenomena exactly, because it uses the full
mathematical formalism: a quantum state of n qubits is a complex vector of
length 2ⁿ, and every gate is a unitary matrix.

  n=1 qubit  →  2  amplitudes  (vector of length 2)
  n=2 qubits →  4  amplitudes  (vector of length 4)
  n=3 qubits →  8  amplitudes  (vector of length 8)
  n=4 qubits → 16  amplitudes  (vector of length 16)

Each amplitude αᵢ is a complex number.  The probability of measuring the
computational basis state |i⟩ is |αᵢ|² and the Born rule requires that
all probabilities sum to 1: Σ|αᵢ|² = 1.

STRUCTURE OF THIS FILE
-----------------------
  1. QuantumState   — represents the state-vector and implements measurement
  2. Gates          — every standard single-qubit and two-qubit gate
  3. Circuit        — assembles gates into a reusable pipeline
  4. Visualiser     — ASCII probability-bar chart (no external GUI needed)
  5. Algorithms     — Bell state, Deutsch-Jozsa, Grover's search
  6. Demo runner    — __main__ block that walks through every concept

DEPENDENCIES
------------
  numpy  — only external dependency; used for complex-vector arithmetic
  matplotlib (optional) — used for a graphical bar chart if available
"""

# ── Standard library ──────────────────────────────────────────────────────────
import random          # for the pseudo-random measurement collapse
import sys             # to inspect runtime environment
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np     # complex arrays and linear algebra

# Try to import matplotlib; the simulator works without it, but if present
# it will also produce a graphical probability bar-chart.
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — QUANTUM STATE
# ══════════════════════════════════════════════════════════════════════════════

class QuantumState:
    """
    Represents the full quantum state of an n-qubit system as a complex
    state-vector (also called a ket, written |ψ⟩ in Dirac notation).

    MATHEMATICS
    -----------
    For n qubits, the state lives in a 2ⁿ-dimensional complex Hilbert space ℂ^(2ⁿ).

    The most general state is a superposition:
        |ψ⟩ = α₀|00…0⟩ + α₁|00…1⟩ + … + α_(2ⁿ-1)|11…1⟩

    where each αᵢ ∈ ℂ and Σᵢ |αᵢ|² = 1  (normalisation).

    Internally we store amplitudes as a NumPy array of dtype complex128.
    Index i of the array corresponds to the computational basis state whose
    binary representation is the n-bit string of i.

    Example — 2 qubits, Bell state |Φ+⟩:
        index 0 → |00⟩,  amplitude = 1/√2
        index 1 → |01⟩,  amplitude = 0
        index 2 → |10⟩,  amplitude = 1/√2
        index 3 → |11⟩,  amplitude = 0

    Qubit ordering convention (IMPORTANT for multi-qubit gates):
        Qubit 0 is the MOST significant bit (leftmost in ket notation).
        So for 3 qubits, index 5 = 0b101 → |101⟩ = qubit0=1, qubit1=0, qubit2=1.
    """

    def __init__(self, n_qubits: int, initial_state: int = 0):
        """
        Initialise the system in a pure computational basis state.

        Parameters
        ----------
        n_qubits    : number of qubits (1–4 supported well; >4 is slow)
        initial_state : integer index of the basis state to start in;
                        default 0 means all qubits in |0⟩.
        """
        if n_qubits < 1 or n_qubits > 10:
            raise ValueError("n_qubits must be between 1 and 10")
        if initial_state < 0 or initial_state >= 2**n_qubits:
            raise ValueError(f"initial_state must be 0..{2**n_qubits - 1}")

        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits   # dimension of the Hilbert space

        # Allocate the state vector with all amplitudes zero …
        self.amplitudes = np.zeros(self.n_states, dtype=complex)
        # … then place amplitude 1 at the chosen basis state.
        # This is a "pure state" — no superposition yet.
        self.amplitudes[initial_state] = 1.0 + 0j

        # Human-readable label for display
        self._label = f"|{initial_state:0{n_qubits}b}⟩"

    # ── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def from_amplitudes(cls, amplitudes: np.ndarray) -> "QuantumState":
        """
        Build a QuantumState directly from an amplitude array.

        The array length must be a power of 2 and the state must be normalised
        (Σ|αᵢ|² ≈ 1).  Useful for setting up exotic initial states in tests.
        """
        n = len(amplitudes)
        if n & (n - 1) != 0:
            raise ValueError("Length of amplitudes must be a power of 2")
        n_qubits = int(np.log2(n))
        qs = cls(n_qubits)          # creates |0…0⟩ temporarily
        qs.amplitudes = np.array(amplitudes, dtype=complex)
        norm = np.linalg.norm(qs.amplitudes)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"State vector must be normalised (norm = {norm:.6f})")
        return qs

    # ── State inspection ─────────────────────────────────────────────────────

    def probabilities(self) -> np.ndarray:
        """
        Return the probability of each computational basis state.

        P(i) = |αᵢ|²

        The Born rule: this is the only way to extract classical information
        from a quantum state.  The act of measurement disturbs the state
        (see `measure` below).
        """
        return np.abs(self.amplitudes) ** 2

    def is_normalised(self, atol: float = 1e-8) -> bool:
        """Sanity check: probabilities must sum to 1."""
        return np.isclose(np.sum(self.probabilities()), 1.0, atol=atol)

    def state_label(self, index: int) -> str:
        """Return the ket label for a basis state index, e.g. index 5 → '|101⟩'."""
        return f"|{index:0{self.n_qubits}b}⟩"

    # ── Measurement ──────────────────────────────────────────────────────────

    def measure(self, shots: int = 1) -> List[int]:
        """
        Simulate projective measurement in the computational basis.

        THEORY
        ------
        Measurement is the irreversible process by which a quantum state
        "collapses" into one of the classical basis states.

        Before measurement: |ψ⟩ = Σ αᵢ|i⟩  — all basis states coexist.
        After  measurement: the system is found in state |k⟩ with probability
                            P(k) = |αₖ|².

        Once measured, the state collapses:  |ψ⟩ → |k⟩.
        Any subsequent measurement on the same collapsed state always gives k.

        Parameters
        ----------
        shots : how many independent measurements to perform.
                Each shot starts from the CURRENT (post-collapse) state.
                For a single shot the state collapses in place.
                For multiple shots the collapse only happens once (first shot)
                and the rest observe the collapsed state — to simulate
                multiple independent preparations, call reset() between shots.

        Returns
        -------
        List of integer outcomes (one per shot).
        """
        probs = self.probabilities()
        outcomes = []

        for _ in range(shots):
            # np.random.choice samples index k with probability probs[k].
            # This faithfully implements the Born rule.
            outcome = int(np.random.choice(self.n_states, p=probs))
            outcomes.append(outcome)

            # State collapse: after measurement the system IS in |outcome⟩.
            # All other amplitudes become zero.
            self.amplitudes = np.zeros(self.n_states, dtype=complex)
            self.amplitudes[outcome] = 1.0 + 0j

            # After collapse all future shots see the same state (|outcome⟩).
            probs = self.probabilities()   # now a delta function at outcome

        return outcomes

    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a SINGLE qubit, leaving the rest of the system in a
        (renormalised) conditional state.

        THEORY
        ------
        Partial measurement:
          1. Compute the marginal probability P(qubit=0) and P(qubit=1).
          2. Sample the outcome.
          3. Project out the other basis states (zero their amplitudes).
          4. Renormalise the surviving amplitudes.

        This is used internally by Grover's algorithm display and can be
        called directly to observe one register of a multi-register system.
        """
        # Build a mask: bit position for this qubit in our MSB-first convention.
        # Qubit 0 → bit position n-1, qubit 1 → bit position n-2, etc.
        bit_position = self.n_qubits - 1 - qubit

        # P(qubit = 1)
        mask_1 = np.array([bool((i >> bit_position) & 1)
                           for i in range(self.n_states)])
        prob_1 = np.sum(self.probabilities()[mask_1])
        prob_0 = 1.0 - prob_1

        # Sample outcome
        outcome = int(np.random.choice([0, 1], p=[prob_0, prob_1]))

        # Project: zero out all amplitudes inconsistent with the outcome
        mask = mask_1 if outcome == 1 else ~mask_1
        new_amp = self.amplitudes.copy()
        new_amp[~mask] = 0.0

        # Renormalise surviving amplitudes
        norm = np.linalg.norm(new_amp)
        if norm > 1e-12:
            new_amp /= norm
        self.amplitudes = new_amp

        return outcome

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"QuantumState ({self.n_qubits} qubits, {self.n_states} basis states)"]
        lines.append("─" * 52)
        probs = self.probabilities()
        for i, (amp, prob) in enumerate(zip(self.amplitudes, probs)):
            if abs(amp) < 1e-9:
                continue   # skip zero-amplitude states to keep output tidy
            label = self.state_label(i)
            # Format amplitude as a + bi
            re, im = amp.real, amp.imag
            amp_str = f"{re:+.4f} {'+' if im >= 0 else '-'} {abs(im):.4f}i"
            bar = "█" * int(prob * 20)
            lines.append(f"  {label}  {amp_str}  P={prob:.4f}  {bar}")
        lines.append("─" * 52)
        lines.append(f"  Σ probabilities = {np.sum(probs):.8f}  "
                     f"({'✓ normalised' if self.is_normalised() else '✗ NOT normalised'})")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — QUANTUM GATES
# ══════════════════════════════════════════════════════════════════════════════

class Gates:
    """
    Factory class for single-qubit and two-qubit gate matrices, plus the
    `apply` helper that embeds a small gate into a full n-qubit unitary.

    THEORY — WHAT IS A QUANTUM GATE?
    ----------------------------------
    A quantum gate is a UNITARY matrix U acting on the state vector:
        |ψ'⟩ = U|ψ⟩

    Unitary means  U†U = I  (U† = conjugate transpose).
    This guarantees:
      • Normalisation is preserved  (gates never violate probability).
      • The operation is reversible (U⁻¹ = U†).

    For a single qubit the gate is 2×2; for n qubits the gate on the FULL
    system is 2ⁿ × 2ⁿ.  When a gate acts on only one qubit in an n-qubit
    system we embed it using the TENSOR PRODUCT (⊗):
        U_full = I ⊗ … ⊗ U ⊗ … ⊗ I
    where U occupies the position of the target qubit.

    ALL GATE MATRICES BELOW ARE WRITTEN IN THE STANDARD COMPUTATIONAL BASIS
    {|0⟩, |1⟩} ordered as column/row 0 = |0⟩, 1 = |1⟩.
    """

    # ── Single-qubit gates ────────────────────────────────────────────────────

    @staticmethod
    def I() -> np.ndarray:
        """
        Identity gate — does nothing.

        Matrix:
            I = [ 1  0 ]
                [ 0  1 ]

        Used as a placeholder when building tensor-product unitaries.
        """
        return np.eye(2, dtype=complex)

    @staticmethod
    def X() -> np.ndarray:
        """
        Pauli-X gate — the quantum NOT gate.

        Matrix:
            X = [ 0  1 ]
                [ 1  0 ]

        Action:
            X|0⟩ = |1⟩    (flips |0⟩ to |1⟩)
            X|1⟩ = |0⟩    (flips |1⟩ to |0⟩)

        This is the quantum analogue of a classical bit-flip.
        Applied to a superposition: X(α|0⟩ + β|1⟩) = α|1⟩ + β|0⟩.
        """
        return np.array([[0, 1],
                         [1, 0]], dtype=complex)

    @staticmethod
    def Y() -> np.ndarray:
        """
        Pauli-Y gate.

        Matrix:
            Y = [ 0  -i ]
                [ i   0 ]

        Action:
            Y|0⟩ =  i|1⟩
            Y|1⟩ = -i|0⟩

        Y combines a bit-flip with a phase flip.
        Together with X and Z it forms the Pauli group, which generates all
        single-qubit unitaries.
        """
        return np.array([[0,  -1j],
                         [1j,   0]], dtype=complex)

    @staticmethod
    def Z() -> np.ndarray:
        """
        Pauli-Z gate — phase flip.

        Matrix:
            Z = [ 1   0 ]
                [ 0  -1 ]

        Action:
            Z|0⟩ =  |0⟩   (|0⟩ unchanged)
            Z|1⟩ = -|1⟩   (|1⟩ gets a global phase of π)

        Z leaves |0⟩ alone but flips the sign of |1⟩.
        Applied to a superposition: Z(α|0⟩ + β|1⟩) = α|0⟩ - β|1⟩.
        This phase difference is invisible to individual measurement but
        affects interference patterns in multi-gate circuits.
        """
        return np.array([[1,  0],
                         [0, -1]], dtype=complex)

    @staticmethod
    def H() -> np.ndarray:
        """
        Hadamard gate — creates and destroys superposition.

        Matrix:
            H = (1/√2) × [ 1   1 ]
                          [ 1  -1 ]

        Action (the most important single-qubit gate):
            H|0⟩ = (|0⟩ + |1⟩)/√2  = |+⟩   (equal superposition)
            H|1⟩ = (|0⟩ - |1⟩)/√2  = |−⟩   (equal superposition with phase)

        Applied twice, H² = I, so H is its own inverse.

        WHY IS H SPECIAL?
        -----------------
        Starting from |0⟩, a single H creates a 50/50 superposition.
        Applied to ALL n qubits of an |0…0⟩ state, H⊗n produces a UNIFORM
        superposition over all 2ⁿ basis states — this is the standard way
        to initialise quantum search algorithms.
        """
        return (1 / np.sqrt(2)) * np.array([[1,  1],
                                            [1, -1]], dtype=complex)

    @staticmethod
    def S() -> np.ndarray:
        """
        S gate (phase gate / √Z).

        Matrix:
            S = [ 1  0 ]
                [ 0  i ]

        Action:
            S|0⟩ = |0⟩
            S|1⟩ = i|1⟩   (adds a phase of π/2 to |1⟩)

        S = Z^(1/2): applying S twice gives Z.
        Used in the quantum Fourier transform and many phase estimation circuits.
        """
        return np.array([[1, 0],
                         [0, 1j]], dtype=complex)

    @staticmethod
    def T() -> np.ndarray:
        """
        T gate (π/8 gate / ⁴√Z).

        Matrix:
            T = [ 1          0      ]
                [ 0   exp(iπ/4)     ]
              = [ 1          0      ]
                [ 0   (1+i)/√2      ]

        Action:
            T|0⟩ = |0⟩
            T|1⟩ = e^(iπ/4)|1⟩   (adds a phase of π/4 to |1⟩)

        T = S^(1/2) = Z^(1/4).  Applying T four times gives Z; twice gives S.

        WHY T IS IMPORTANT
        ------------------
        {H, T} together form a UNIVERSAL gate set: any unitary on any number
        of qubits can be approximated to arbitrary precision using only H and
        T gates (Solovay–Kitaev theorem).  T is the 'non-Clifford' gate that
        provides the extra power beyond the Clifford group.
        """
        return np.array([[1,                0],
                         [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """
        Rotation around the X-axis of the Bloch sphere by angle theta.

            Rx(θ) = exp(-i θ X / 2)
                  = [ cos(θ/2)    -i·sin(θ/2) ]
                    [ -i·sin(θ/2)  cos(θ/2)   ]

        At θ=π this equals -iX (equivalent to X up to global phase).
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c,     -1j * s],
                         [-1j * s,  c  ]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """
        Rotation around the Y-axis of the Bloch sphere by angle theta.

            Ry(θ) = [ cos(θ/2)  -sin(θ/2) ]
                    [ sin(θ/2)   cos(θ/2) ]
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s],
                         [s,  c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """
        Rotation around the Z-axis of the Bloch sphere by angle theta.

            Rz(θ) = [ exp(-iθ/2)       0      ]
                    [     0        exp(+iθ/2)  ]
        """
        return np.array([[np.exp(-1j * theta / 2),              0],
                         [0,             np.exp(+1j * theta / 2)]], dtype=complex)

    @staticmethod
    def Phase(phi: float) -> np.ndarray:
        """
        General phase gate — adds phase e^(iφ) to |1⟩.

            P(φ) = [ 1       0    ]
                   [ 0   e^(iφ)   ]

        Special cases:
            P(π)   = Z
            P(π/2) = S
            P(π/4) = T
        """
        return np.array([[1,                 0],
                         [0, np.exp(1j * phi)]], dtype=complex)

    # ── Two-qubit gates ───────────────────────────────────────────────────────

    @staticmethod
    def CNOT() -> np.ndarray:
        """
        Controlled-NOT (CX) gate — THE entangling gate.

        Acts on 2 qubits: a CONTROL qubit and a TARGET qubit.
        The convention here is qubit 0 = control, qubit 1 = target.

        Matrix (in the 2-qubit basis |00⟩, |01⟩, |10⟩, |11⟩):

            CNOT = [ 1  0  0  0 ]
                   [ 0  1  0  0 ]
                   [ 0  0  0  1 ]
                   [ 0  0  1  0 ]

        Action:
            |00⟩ → |00⟩   (control=0, target unchanged)
            |01⟩ → |01⟩   (control=0, target unchanged)
            |10⟩ → |11⟩   (control=1, flip target: 0→1)
            |11⟩ → |10⟩   (control=1, flip target: 1→0)

        WHY CNOT CREATES ENTANGLEMENT
        ------------------------------
        Input  |+0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩+|10⟩)/√2
        After CNOT: (|00⟩+|11⟩)/√2  — this is the Bell state |Φ+⟩.
        This state CANNOT be written as a product α|q0⟩ ⊗ β|q1⟩, so the
        two qubits are entangled.
        """
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)

    @staticmethod
    def CZ() -> np.ndarray:
        """
        Controlled-Z gate.

        Applies Z to the target qubit only when the control qubit is |1⟩.

            CZ = [ 1  0  0   0 ]
                 [ 0  1  0   0 ]
                 [ 0  0  1   0 ]
                 [ 0  0  0  -1 ]

        Action:
            |00⟩ → |00⟩
            |01⟩ → |01⟩
            |10⟩ → |10⟩
            |11⟩ → -|11⟩   (phase flip only when both qubits are |1⟩)

        CZ is symmetric: qubit 0 and qubit 1 play equal roles.
        """
        return np.array([[1, 0, 0,  0],
                         [0, 1, 0,  0],
                         [0, 0, 1,  0],
                         [0, 0, 0, -1]], dtype=complex)

    @staticmethod
    def SWAP() -> np.ndarray:
        """
        SWAP gate — exchanges the states of two qubits.

            SWAP = [ 1  0  0  0 ]
                   [ 0  0  1  0 ]
                   [ 0  1  0  0 ]
                   [ 0  0  0  1 ]

        Action:
            |00⟩ → |00⟩
            |01⟩ → |10⟩   (01 ↔ 10)
            |10⟩ → |01⟩
            |11⟩ → |11⟩

        SWAP can be decomposed into three CNOT gates:
            SWAP = CNOT(0→1) · CNOT(1→0) · CNOT(0→1)
        """
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)

    @staticmethod
    def Toffoli() -> np.ndarray:
        """
        Toffoli gate (CCNOT / controlled-controlled-NOT).

        A 3-qubit gate: flips the TARGET qubit only when BOTH control qubits
        are |1⟩.

        This is the quantum analogue of the classical NAND/AND gate and is
        UNIVERSAL for classical reversible computation.

        Matrix is 8×8 (3 qubits → 8 basis states).  It is the identity on all
        states except |110⟩ ↔ |111⟩:
            |110⟩ → |111⟩
            |111⟩ → |110⟩
        """
        mat = np.eye(8, dtype=complex)
        # Swap rows/cols 6 (|110⟩) and 7 (|111⟩)
        mat[6, 6] = 0;  mat[6, 7] = 1
        mat[7, 6] = 1;  mat[7, 7] = 0
        return mat

    # ── Gate application engine ───────────────────────────────────────────────

    @staticmethod
    def apply_single(state: QuantumState, gate: np.ndarray, qubit: int) -> None:
        """
        Apply a 2×2 single-qubit gate to one qubit of an n-qubit state.

        IMPLEMENTATION STRATEGY
        -----------------------
        We do NOT explicitly build the full 2ⁿ × 2ⁿ unitary via tensor products
        (that would allocate huge matrices).  Instead, we reshape the amplitude
        vector into a tensor with shape (2, 2, …, 2)  — one axis per qubit —
        and use np.tensordot to contract the gate matrix along the target axis.
        Then we move the result axis back to the correct position.

        This is equivalent to:
            U_full = I ⊗ … ⊗ gate ⊗ … ⊗ I   (gate at position `qubit`)
        but without materialising the exponentially large U_full.

        Parameters
        ----------
        state : QuantumState to modify IN PLACE
        gate  : 2×2 unitary matrix
        qubit : index of the qubit to act on (0 = most significant)
        """
        n = state.n_qubits

        # Reshape: (2ⁿ,) → (2, 2, …, 2)
        tensor = state.amplitudes.reshape([2] * n)

        # tensordot contracts axis `qubit` of `tensor` with axis 1 of `gate`.
        # Result shape: (2, then all other qubit axes) — gate output axis first.
        result = np.tensordot(gate, tensor, axes=([1], [qubit]))
        # result.shape = (2, 2, …, 2) but with the gate axis at position 0.

        # Move the contracted axis back to position `qubit`.
        result = np.moveaxis(result, 0, qubit)

        # Flatten back to 1D and store.
        state.amplitudes = result.reshape(state.n_states)

    @staticmethod
    def apply_two(state: QuantumState, gate: np.ndarray,
                  qubit0: int, qubit1: int) -> None:
        """
        Apply a 4×4 two-qubit gate to two qubits of an n-qubit state.

        Works by reshaping the state vector and contracting with the gate
        tensor.  The gate's 4×4 matrix is reshaped to (2,2,2,2) and the
        contraction runs over the two target qubit axes simultaneously.

        Parameters
        ----------
        state  : QuantumState to modify IN PLACE
        gate   : 4×4 unitary matrix (ordered: qubit0 MSB, qubit1 LSB)
        qubit0 : index of the first (most-significant / control) qubit
        qubit1 : index of the second (least-significant / target) qubit
        """
        n = state.n_qubits

        # Reshape gate from (4,4) to (2,2,2,2):
        #   axis 0 = output qubit0, axis 1 = output qubit1
        #   axis 2 = input  qubit0, axis 3 = input  qubit1
        gate_tensor = gate.reshape([2, 2, 2, 2])

        # Reshape state from (2ⁿ,) to (2,2,…,2)
        tensor = state.amplitudes.reshape([2] * n)

        # Contract over the two input axes (axes 2,3 of gate vs qubit0, qubit1 of state)
        result = np.tensordot(gate_tensor, tensor, axes=([2, 3], [qubit0, qubit1]))
        # result.shape: (2, 2, <remaining qubit axes>)
        # Axes 0 and 1 are the two output qubit axes; they need to go back to qubit0, qubit1.

        # Currently axes are: 0→out_q0, 1→out_q1, then all n qubits except qubit0,qubit1.
        # We need to move axes 0 and 1 back to positions qubit0 and qubit1.
        # The remaining qubit axes appear in their original order but with qubit0,qubit1 gaps.

        remaining_axes = [q for q in range(n) if q not in (qubit0, qubit1)]
        # result axes: 0, 1, then remaining_axes[0], remaining_axes[1], ...
        # Build final axis order: insert out_q0 at qubit0, out_q1 at qubit1
        final_order = list(range(2, 2 + len(remaining_axes)))  # remaining axes' current positions
        # Insert output axes into the correct positions
        for pos, axis in zip(sorted([qubit0, qubit1]), [0, 1]):
            final_order.insert(pos, axis)
        result = np.transpose(result, final_order)

        state.amplitudes = result.reshape(state.n_states)

    @staticmethod
    def apply_three(state: QuantumState, gate: np.ndarray,
                    q0: int, q1: int, q2: int) -> None:
        """
        Apply an 8×8 three-qubit gate (e.g. Toffoli) to three qubits.

        Same tensor-contraction strategy as apply_two, extended to 3 qubits.
        """
        n = state.n_qubits
        gate_tensor = gate.reshape([2, 2, 2, 2, 2, 2])
        tensor = state.amplitudes.reshape([2] * n)

        result = np.tensordot(gate_tensor, tensor, axes=([3, 4, 5], [q0, q1, q2]))

        remaining = [q for q in range(n) if q not in (q0, q1, q2)]
        final_order = list(range(3, 3 + len(remaining)))
        for pos, axis in zip(sorted([q0, q1, q2]), [0, 1, 2]):
            final_order.insert(pos, axis)
        result = np.transpose(result, final_order)

        state.amplitudes = result.reshape(state.n_states)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QUANTUM CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

class QuantumCircuit:
    """
    A sequential pipeline of quantum gates applied to a QuantumState.

    USAGE
    -----
        qc = QuantumCircuit(2)
        qc.h(0)          # Hadamard on qubit 0
        qc.cnot(0, 1)    # CNOT with control=0, target=1
        state = qc.run()
        print(state)

    INTERNAL REPRESENTATION
    -----------------------
    Each gate is stored as a tuple: (name, gate_matrix, [qubit_indices...])
    When `run()` is called the gates are applied left-to-right (earliest first).
    """

    def __init__(self, n_qubits: int, initial_state: int = 0):
        self.n_qubits = n_qubits
        self.initial_state = initial_state
        self._ops: List[Tuple] = []          # list of (label, matrix, qubits)
        self._history: List[str] = []        # human-readable gate log

    # ── Gate-adding methods ───────────────────────────────────────────────────

    def _add(self, label: str, matrix: np.ndarray, *qubits: int):
        """Internal: record a gate operation."""
        self._ops.append((label, matrix, list(qubits)))
        qubit_str = ", ".join(str(q) for q in qubits)
        self._history.append(f"{label}({qubit_str})")

    def i(self, qubit: int):   self._add("I",     Gates.I(),   qubit)
    def x(self, qubit: int):   self._add("X",     Gates.X(),   qubit)
    def y(self, qubit: int):   self._add("Y",     Gates.Y(),   qubit)
    def z(self, qubit: int):   self._add("Z",     Gates.Z(),   qubit)
    def h(self, qubit: int):   self._add("H",     Gates.H(),   qubit)
    def s(self, qubit: int):   self._add("S",     Gates.S(),   qubit)
    def t(self, qubit: int):   self._add("T",     Gates.T(),   qubit)

    def rx(self, qubit: int, theta: float):
        self._add(f"Rx({theta:.3f})", Gates.Rx(theta), qubit)

    def ry(self, qubit: int, theta: float):
        self._add(f"Ry({theta:.3f})", Gates.Ry(theta), qubit)

    def rz(self, qubit: int, theta: float):
        self._add(f"Rz({theta:.3f})", Gates.Rz(theta), qubit)

    def phase(self, qubit: int, phi: float):
        self._add(f"P({phi:.3f})", Gates.Phase(phi), qubit)

    def cnot(self, control: int, target: int):
        self._add("CNOT", Gates.CNOT(), control, target)

    def cz(self, qubit0: int, qubit1: int):
        self._add("CZ", Gates.CZ(), qubit0, qubit1)

    def swap(self, qubit0: int, qubit1: int):
        self._add("SWAP", Gates.SWAP(), qubit0, qubit1)

    def toffoli(self, c0: int, c1: int, target: int):
        self._add("Toffoli", Gates.Toffoli(), c0, c1, target)

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, initial_state: Optional[int] = None) -> QuantumState:
        """
        Execute all gates in sequence and return the final QuantumState.

        A fresh QuantumState is created at the start of each run so the
        circuit is reusable (no hidden state between calls).
        """
        init = initial_state if initial_state is not None else self.initial_state
        state = QuantumState(self.n_qubits, init)

        for label, matrix, qubits in self._ops:
            n_q = len(qubits)
            if n_q == 1:
                Gates.apply_single(state, matrix, qubits[0])
            elif n_q == 2:
                Gates.apply_two(state, matrix, qubits[0], qubits[1])
            elif n_q == 3:
                Gates.apply_three(state, matrix, qubits[0], qubits[1], qubits[2])
            else:
                raise NotImplementedError(f"No applicator for {n_q}-qubit gates")

        return state

    def diagram(self) -> str:
        """Return a simple text diagram of the circuit."""
        lines = [f"Circuit ({self.n_qubits} qubits):"]
        lines.append("  " + " → ".join(self._history) if self._history else "  (empty)")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISER
# ══════════════════════════════════════════════════════════════════════════════

class Visualiser:
    """
    Renders the probability distribution of a QuantumState as a bar chart.

    ASCII mode  — always available, prints to the terminal.
    Matplotlib  — produces a graphical chart if matplotlib is installed.
    """

    BAR_WIDTH = 40   # characters in a full bar (probability = 1.0)

    @staticmethod
    def ascii_bar(state: QuantumState, title: str = "") -> str:
        """
        Return an ASCII probability histogram as a string.

        Example output for a 2-qubit Bell state:
            |00⟩  ████████████████████  50.00%
            |01⟩                         0.00%
            |10⟩  ████████████████████  50.00%
            |11⟩                         0.00%
        """
        probs = state.probabilities()
        width = Visualiser.BAR_WIDTH

        lines = []
        if title:
            lines.append(f"\n{'═' * (width + 20)}")
            lines.append(f"  {title}")
            lines.append(f"{'═' * (width + 20)}")

        for i, prob in enumerate(probs):
            label = state.state_label(i)
            filled = int(round(prob * width))
            bar = "█" * filled + "░" * (width - filled)
            lines.append(f"  {label}  {bar}  {prob * 100:6.2f}%")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def plot(state: QuantumState, title: str = "", save_path: Optional[str] = None):
        """
        Display a matplotlib bar chart of the probability distribution.

        Falls back to ASCII if matplotlib is unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            print(Visualiser.ascii_bar(state, title))
            return

        probs = state.probabilities()
        labels = [state.state_label(i) for i in range(state.n_states)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: probability bar chart
        colors = plt.cm.viridis(probs / (probs.max() + 1e-12))
        bars = axes[0].bar(labels, probs, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Probability  |α|²")
        axes[0].set_xlabel("Computational basis state")
        axes[0].set_title(f"Probability Distribution\n{title}")
        axes[0].tick_params(axis="x", rotation=45)
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                axes[0].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.01,
                             f"{prob:.3f}",
                             ha="center", va="bottom", fontsize=8)

        # Right: amplitude Argand diagram (real vs imaginary parts)
        re = state.amplitudes.real
        im = state.amplitudes.imag
        sc = axes[1].scatter(re, im, c=range(len(re)),
                             cmap="tab20", s=100, edgecolors="black", linewidth=0.5)
        for i, (r, img) in enumerate(zip(re, im)):
            if abs(state.amplitudes[i]) > 1e-9:
                axes[1].annotate(labels[i], (r, img),
                                 textcoords="offset points", xytext=(5, 5), fontsize=8)
        # Unit circle for reference
        theta_c = np.linspace(0, 2 * np.pi, 300)
        axes[1].plot(np.cos(theta_c), np.sin(theta_c), "k--", alpha=0.2, linewidth=0.8)
        axes[1].axhline(0, color="grey", linewidth=0.5)
        axes[1].axvline(0, color="grey", linewidth=0.5)
        axes[1].set_xlim(-1.2, 1.2);  axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("Re(α)");   axes[1].set_ylabel("Im(α)")
        axes[1].set_title("Amplitude in ℂ (Argand diagram)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [Saved chart to {save_path}]")
        else:
            plt.show()
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUANTUM ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

class QuantumAlgorithms:
    """
    A collection of landmark quantum algorithms implemented using
    QuantumCircuit.  Each method returns a (circuit, state, description)
    tuple and is extensively annotated.
    """

    # ── Bell States ────────────────────────────────────────────────────────────

    @staticmethod
    def bell_state(which: str = "phi+") -> Tuple[QuantumCircuit, QuantumState, str]:
        """
        Prepare one of the four Bell (EPR) states — the maximally entangled
        2-qubit states.

        THEORY
        ------
        The four Bell states are:
            |Φ+⟩ = (|00⟩ + |11⟩)/√2    'phi+'  (default)
            |Φ-⟩ = (|00⟩ - |11⟩)/√2    'phi-'
            |Ψ+⟩ = (|01⟩ + |10⟩)/√2    'psi+'
            |Ψ-⟩ = (|01⟩ - |10⟩)/√2    'psi-'

        They form an orthonormal basis for the 4-dimensional 2-qubit Hilbert
        space — the "Bell basis".

        CIRCUIT FOR |Φ+⟩
        ─────────────────
          q0: ─ H ─●─
                    │
          q1: ─────X─

        Step 1: H on q0 creates  (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
        Step 2: CNOT flips q1 when q0=1:          (|00⟩ + |11⟩)/√2

        ENTANGLEMENT TEST
        -----------------
        After preparing |Φ+⟩, measure q0.
          • If q0 = 0, q1 collapses to |0⟩ — measure q1 → always 0.
          • If q0 = 1, q1 collapses to |1⟩ — measure q1 → always 1.
        This perfect correlation holds even if the qubits are separated by
        any distance (Einstein's "spooky action at a distance").
        """
        variants = {
            "phi+": (0, False, False),   # (initial_q1_flip, Z_on_q0, Z_on_q1)
            "phi-": (0, True,  False),
            "psi+": (1, False, False),
            "psi-": (1, True,  False),
        }
        if which not in variants:
            raise ValueError(f"Bell state must be one of {list(variants.keys())}")

        q1_flip, z_q0, _ = variants[which]

        qc = QuantumCircuit(2)
        if q1_flip:
            qc.x(1)    # set q1 to |1⟩ for Ψ states
        qc.h(0)        # put q0 in superposition
        if z_q0:
            qc.z(0)    # add phase for - variants
        qc.cnot(0, 1)  # entangle

        state = qc.run()
        desc = (f"Bell state |{which.replace('phi', 'Φ').replace('psi', 'Ψ')}⟩\n"
                f"  Circuit: {qc.diagram()}\n"
                f"  Maximally entangled: measuring either qubit collapses both.")
        return qc, state, desc

    # ── GHZ State ─────────────────────────────────────────────────────────────

    @staticmethod
    def ghz_state(n_qubits: int = 3) -> Tuple[QuantumCircuit, QuantumState, str]:
        """
        Greenberger–Horne–Zeilinger (GHZ) state — maximum multi-qubit entanglement.

        THEORY
        ------
        The n-qubit GHZ state is:
            |GHZ_n⟩ = (|00…0⟩ + |11…1⟩)/√2

        For n=3: |GHZ⟩ = (|000⟩ + |111⟩)/√2

        It is an n-way generalisation of the Bell state |Φ+⟩.  All n qubits
        are perfectly correlated: measuring any one qubit determines all others.

        CIRCUIT
        -------
          q0: ─ H ─●───────
                    │
          q1: ─────X─●─────
                      │
          q2: ─────────X───
          (and so on for more qubits)
        """
        if n_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits")

        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cnot(i, i + 1)

        state = qc.run()
        desc = (f"{n_qubits}-qubit GHZ state: (|{'0'*n_qubits}⟩ + |{'1'*n_qubits}⟩)/√2\n"
                f"  All {n_qubits} qubits maximally entangled.")
        return qc, state, desc

    # ── Deutsch-Jozsa Algorithm ────────────────────────────────────────────────

    @staticmethod
    def deutsch_jozsa(n_input: int = 2,
                      oracle_type: str = "balanced"
                      ) -> Tuple[QuantumCircuit, QuantumState, str]:
        """
        Deutsch-Jozsa algorithm — exponential quantum speedup over classical.

        PROBLEM STATEMENT
        ------------------
        Given a black-box function f: {0,1}ⁿ → {0,1} that is EITHER:
          • CONSTANT  — f(x) is the same for all x  (all 0 or all 1), OR
          • BALANCED  — f(x) = 0 for exactly half the inputs, 1 for the other half.

        Determine which case applies.

        Classical: needs up to 2^(n-1)+1 evaluations in the worst case.
        Quantum  : needs exactly ONE evaluation of the oracle (exponential speedup).

        HOW IT WORKS
        ------------
        Circuit for n=2 input qubits + 1 ancilla qubit:

          q0: ─ H ──────[Oracle]── H ──M
          q1: ─ H ──────[Oracle]── H ──M
          q2: ─ X ─ H ──[Oracle]──────

        1. Initialise: n input qubits in |0⟩, ancilla in |1⟩.
        2. H on ALL qubits → input in equal superposition, ancilla in |−⟩=(|0⟩-|1⟩)/√2.
        3. Apply oracle Uf (encodes f as a phase kick-back).
        4. H on input qubits.
        5. Measure input qubits.
           • All zeros → f is CONSTANT.
           • Any non-zero → f is BALANCED.

        PHASE KICK-BACK TRICK
        ---------------------
        When the ancilla is in |−⟩ = (|0⟩-|1⟩)/√2, applying the oracle
        Uf|x⟩|−⟩ = (-1)^f(x)|x⟩|−⟩  leaves the ancilla unchanged but
        encodes f(x) as a ±1 phase on the input register.

        Parameters
        ----------
        n_input     : number of input qubits (total qubits = n_input + 1)
        oracle_type : 'constant_0', 'constant_1', or 'balanced'
        """
        n_total = n_input + 1                  # +1 for ancilla qubit
        ancilla = n_input                       # ancilla is the LAST qubit

        qc = QuantumCircuit(n_total)

        # Step 1: put ancilla in |1⟩
        qc.x(ancilla)

        # Step 2: H on all qubits
        for i in range(n_total):
            qc.h(i)

        # Step 3: Oracle (Uf)
        if oracle_type == "constant_0":
            pass   # f(x)=0 → identity oracle (no gates needed)

        elif oracle_type == "constant_1":
            # f(x)=1 for all x → flip ancilla for every input
            # Phase kick-back: applying X to ancilla when f(x)=1 for ALL x
            # is equivalent to a global -1 phase, undetectable in measurement.
            qc.x(ancilla)

        elif oracle_type == "balanced":
            # f(x) = x₀ XOR x₁ (for n=2); this is balanced.
            # Oracle: CNOT from each input qubit to ancilla.
            for i in range(n_input):
                qc.cnot(i, ancilla)
        else:
            raise ValueError("oracle_type must be 'constant_0', 'constant_1', or 'balanced'")

        # Step 4: H on input qubits (NOT ancilla)
        for i in range(n_input):
            qc.h(i)

        # Run circuit
        state = qc.run()
        probs = state.probabilities()

        # The input register lives in qubits 0..n_input-1.
        # We check: is probability concentrated on |00…0⟩ (first n_input bits = 0)?
        # Sum probabilities where all input qubits are 0.
        prob_all_zeros = sum(
            probs[i]
            for i in range(state.n_states)
            if (i >> 1) == 0           # top n_input bits are 0
        )
        conclusion = "CONSTANT" if prob_all_zeros > 0.9 else "BALANCED"

        desc = (
            f"Deutsch-Jozsa  ({n_input} input qubits, oracle='{oracle_type}')\n"
            f"  One query suffices to distinguish constant vs balanced f.\n"
            f"  Result: f is {conclusion}  "
            f"({'correct ✓' if conclusion.lower() == oracle_type.replace('constant_0','constant').replace('constant_1','constant') else 'check oracle'})."
        )
        return qc, state, desc

    # ── Grover's Search Algorithm ──────────────────────────────────────────────

    @staticmethod
    def grover(n_qubits: int = 2,
               target: int = 2) -> Tuple[QuantumCircuit, QuantumState, str]:
        """
        Grover's quantum search algorithm — quadratic speedup.

        PROBLEM STATEMENT
        ------------------
        Given an unstructured database of N = 2ⁿ items, find the one item
        that satisfies a given predicate (the "marked" item).

        Classical : needs O(N) queries on average.
        Quantum   : needs O(√N) queries — a QUADRATIC speedup.

        HOW IT WORKS
        ------------
        Starting state: uniform superposition |s⟩ = H^⊗n|0…0⟩ = (1/√N) Σ|x⟩

        Repeat ~π√N/4 times:
          1. ORACLE Oₜ  : flips the phase of the marked state |t⟩.
                          |x⟩ → (-1)^δ(x,t)|x⟩
          2. DIFFUSION D : inversion about the mean (amplifies the marked state).
                          D = 2|s⟩⟨s| - I

        After ~π√N/4 iterations, the marked state has amplitude ≈ 1.
        Measuring then finds the target with high probability.

        GEOMETRIC INTUITION
        -------------------
        The initial state |s⟩ lies close to the subspace orthogonal to |t⟩.
        Each oracle + diffusion step rotates |s⟩ by an angle 2θ toward |t⟩,
        where sin(θ) = 1/√N.  After π/(4θ) ≈ π√N/4 steps, |s⟩ ≈ |t⟩.

        CIRCUIT
        --------
        For n=2, target=2 (|10⟩), 1 iteration:

          q0: ─ H ─[Oracle]─ H ─ X ─●─ X ─ H ─
                                      │
          q1: ─ H ─[Oracle]─ H ─ X ─ Z ─ X ─ H ─

        The diffusion operator D = H^⊗n (2|0⟩⟨0| - I) H^⊗n is implemented as:
          H^⊗n → X^⊗n → CZ (or multi-controlled Z) → X^⊗n → H^⊗n

        Parameters
        ----------
        n_qubits : number of qubits (N = 2ⁿ items)
        target   : integer index of the marked item (0 ≤ target < 2ⁿ)
        """
        if target >= 2 ** n_qubits:
            raise ValueError(f"target must be < {2**n_qubits}")

        N = 2 ** n_qubits
        n_iterations = max(1, int(round(np.pi / 4 * np.sqrt(N))))

        qc = QuantumCircuit(n_qubits)

        # Step 1: Create uniform superposition
        for i in range(n_qubits):
            qc.h(i)

        for _ in range(n_iterations):
            # ── Oracle: flip phase of |target⟩ ────────────────────────────
            # Strategy: X gates to flip qubits where the target bit is 0,
            # then apply multi-controlled Z (implemented via CZ for n=2 or
            # via decomposition for higher n), then X gates to undo the flips.

            target_bits = [(target >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]

            # Flip qubits where target has a 0
            for i, bit in enumerate(target_bits):
                if bit == 0:
                    qc.x(i)

            # Apply multi-controlled Z
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                # CZ decomposes as: H(q1), CNOT(q0→q1), H(q1)
                qc.h(1)
                qc.cnot(0, 1)
                qc.h(1)
            elif n_qubits == 3:
                # Toffoli + H on target qubit implements CCZ
                qc.h(2)
                qc.toffoli(0, 1, 2)
                qc.h(2)
            else:
                # For 4 qubits: decompose into Toffoli gates
                # CCZ on qubits 0,1,2 then CZ on result with qubit 3
                # (simplified — using ancilla-free decomposition)
                qc.h(3)
                qc.toffoli(0, 1, 2)
                qc.cnot(2, 3)
                qc.toffoli(0, 1, 2)
                qc.h(3)

            # Undo the 0-qubit flips
            for i, bit in enumerate(target_bits):
                if bit == 0:
                    qc.x(i)

            # ── Diffusion operator: 2|s⟩⟨s| - I ──────────────────────────
            # Implemented as H^⊗n  (2|0⟩⟨0| - I)  H^⊗n

            # H on all qubits
            for i in range(n_qubits):
                qc.h(i)

            # 2|0⟩⟨0| - I = X^⊗n (2|1…1⟩⟨1…1| - I) X^⊗n
            # First flip all qubits
            for i in range(n_qubits):
                qc.x(i)

            # Multi-controlled phase flip of |1…1⟩
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                qc.h(1)
                qc.cnot(0, 1)
                qc.h(1)
            elif n_qubits == 3:
                qc.h(2)
                qc.toffoli(0, 1, 2)
                qc.h(2)
            else:
                qc.h(3)
                qc.toffoli(0, 1, 2)
                qc.cnot(2, 3)
                qc.toffoli(0, 1, 2)
                qc.h(3)

            # Undo the all-flip
            for i in range(n_qubits):
                qc.x(i)

            # H on all qubits (complete diffusion)
            for i in range(n_qubits):
                qc.h(i)

        state = qc.run()
        probs = state.probabilities()
        found = int(np.argmax(probs))
        success = found == target

        desc = (
            f"Grover's Search  ({n_qubits} qubits, N={N} items, target={target} = "
            f"|{target:0{n_qubits}b}⟩, {n_iterations} iteration(s))\n"
            f"  Classical: O({N}) queries.  Quantum: O({int(np.sqrt(N))}) queries.\n"
            f"  Highest-probability state: |{found:0{n_qubits}b}⟩  "
            f"(P={probs[found]:.3f})  {'✓ correct' if success else '✗ missed'}"
        )
        return qc, state, desc

    # ── Quantum Teleportation Protocol ────────────────────────────────────────

    @staticmethod
    def teleportation() -> Tuple[QuantumCircuit, QuantumState, str]:
        """
        Quantum teleportation — transfer of an unknown qubit state using
        entanglement and 2 classical bits.

        PROTOCOL  (3 qubits: q0=message, q1=Alice's EPR half, q2=Bob's EPR half)
        ---------
        1. Prepare an arbitrary message state on q0: |ψ⟩ = α|0⟩ + β|1⟩.
           Here we use Ry(π/3) on |0⟩ as the message.
        2. Create Bell pair (q1, q2): H(q1), CNOT(q1→q2).
        3. Alice's joint measurement: CNOT(q0→q1), H(q0).
        4. Alice sends 2 classical bits (results of measuring q0 and q1) to Bob.
        5. Bob applies corrections:
             if classical bit from q1 = 1 → apply X to q2
             if classical bit from q0 = 1 → apply Z to q2
           After corrections, q2 is in state |ψ⟩.

        NOTE: We simulate steps 1–3 (the entanglement and Bell-basis rotation).
        The actual measurement and classical feed-forward would collapse the state.
        Instead we show the joint state before Alice measures.
        """
        qc = QuantumCircuit(3)

        # Prepare message qubit q0 in an arbitrary state |ψ⟩ = Ry(π/3)|0⟩
        # This gives |ψ⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩
        qc.ry(0, np.pi / 3)

        # Create Bell pair between q1 (Alice) and q2 (Bob)
        qc.h(1)
        qc.cnot(1, 2)

        # Alice's Bell measurement preparation
        qc.cnot(0, 1)
        qc.h(0)

        state = qc.run()
        desc = (
            "Quantum Teleportation (3 qubits)\n"
            "  q0 = message: Ry(π/3)|0⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩\n"
            "  q1 = Alice's entangled qubit,  q2 = Bob's entangled qubit\n"
            "  State shown: after Alice's Bell-measurement preparation.\n"
            "  Alice measures q0,q1 → 2 classical bits → Bob applies X/Z to q2."
        )
        return qc, state, desc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DEMO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    """Print a bold section header."""
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def demo_single_qubit():
    """Walk through every single-qubit gate on a 1-qubit system."""
    section("DEMO 1 — Single Qubit: Basis, Superposition, Phase, Measurement")

    print("""
  A single qubit is described by a 2-element complex vector:
      |ψ⟩ = α|0⟩ + β|1⟩,   where |α|² + |β|² = 1

  |0⟩ and |1⟩ are the computational basis states (like classical 0 and 1).
  The numbers α and β are complex AMPLITUDES; their squared magnitudes give
  the probabilities of measuring 0 or 1 respectively.
    """)

    # ── Initial state |0⟩ ────────────────────────────────────────────────────
    print("  ┌─ Initial state |0⟩ ──────────────────────────────────────────┐")
    q = QuantumState(1)
    print(q)
    print(Visualiser.ascii_bar(q, "|0⟩"))

    # ── X gate (NOT) ─────────────────────────────────────────────────────────
    print("  ┌─ After X gate (NOT): |0⟩ → |1⟩ ─────────────────────────────┐")
    q = QuantumState(1)
    Gates.apply_single(q, Gates.X(), 0)
    print(q)
    print(Visualiser.ascii_bar(q, "X|0⟩ = |1⟩"))

    # ── H gate (superposition) ─────────────────────────────────────────────
    print("  ┌─ After H gate: |0⟩ → (|0⟩+|1⟩)/√2 (equal superposition) ───┐")
    q = QuantumState(1)
    Gates.apply_single(q, Gates.H(), 0)
    print(q)
    print(Visualiser.ascii_bar(q, "H|0⟩ = |+⟩"))

    print("  H twice = Identity (H is its own inverse):")
    Gates.apply_single(q, Gates.H(), 0)
    print(q)

    # ── Z gate (phase flip) ────────────────────────────────────────────────
    print("  ┌─ Z gate on |+⟩: creates |−⟩ = (|0⟩−|1⟩)/√2 ────────────────┐")
    q = QuantumState(1)
    Gates.apply_single(q, Gates.H(), 0)
    Gates.apply_single(q, Gates.Z(), 0)
    print(q)
    print(Visualiser.ascii_bar(q, "ZH|0⟩ = |−⟩"))

    # ── S and T gates ─────────────────────────────────────────────────────
    print("  ┌─ S gate (π/2 phase) and T gate (π/4 phase) on |+⟩ ──────────┐")
    for gate_fn, name, phase in [(Gates.S, "S", "π/2"), (Gates.T, "T", "π/4")]:
        q = QuantumState(1)
        Gates.apply_single(q, Gates.H(), 0)
        Gates.apply_single(q, gate_fn(), 0)
        print(f"  {name} gate (adds e^(i{phase}) to |1⟩ amplitude):")
        print(q)

    # ── Measurement collapse ───────────────────────────────────────────────
    print("  ┌─ Measurement of superposition state ──────────────────────────┐")
    print("""
  We prepare |+⟩ = (|0⟩+|1⟩)/√2 and measure 20 times.
  Each run gives a fresh superposition (we do NOT collapse between shots).
  Expected: ~10 zeros, ~10 ones.
    """)
    results = []
    for _ in range(20):
        q = QuantumState(1)
        Gates.apply_single(q, Gates.H(), 0)
        outcome = q.measure(shots=1)[0]
        results.append(outcome)
    print(f"  20 measurements of |+⟩: {results}")
    print(f"  Count(0)={results.count(0)}, Count(1)={results.count(1)}")


def demo_two_qubit():
    """Demonstrate 2-qubit systems, product states, and entanglement."""
    section("DEMO 2 — Two Qubits: Product States, CNOT, Entanglement")

    print("""
  A 2-qubit system has 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩.
  The state vector has 4 complex amplitudes.

  A PRODUCT STATE is one that can be written as q0⊗q1.
  An ENTANGLED STATE cannot — measuring one qubit determines the other.
    """)

    # ── Product state ────────────────────────────────────────────────────────
    print("  ┌─ Product state: H on q0, identity on q1 ──────────────────────┐")
    q = QuantumState(2)
    Gates.apply_single(q, Gates.H(), 0)
    print(q)
    print(Visualiser.ascii_bar(q, "(|0⟩+|1⟩)/√2 ⊗ |0⟩"))

    # ── Bell state |Φ+⟩ ──────────────────────────────────────────────────────
    print("  ┌─ Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 ──────────────────────────┐")
    _, state, desc = QuantumAlgorithms.bell_state("phi+")
    print(f"  {desc}")
    print(state)
    print(Visualiser.ascii_bar(state, "Bell state |Φ+⟩"))

    # ── Correlation demo ──────────────────────────────────────────────────────
    print("  ┌─ Entanglement correlation: measure q0, observe q1 ────────────┐")
    print("""
  We prepare |Φ+⟩ 10 times and measure qubit 0, then qubit 1.
  Perfect entanglement predicts: outcomes always agree (both 0 or both 1).
    """)
    for trial in range(10):
        _, s, _ = QuantumAlgorithms.bell_state("phi+")
        r0 = s.measure_qubit(0)
        r1 = s.measure_qubit(1)
        match = "✓" if r0 == r1 else "✗"
        print(f"    Trial {trial+1:2d}: q0={r0}, q1={r1}  {match}")

    # ── All four Bell states ───────────────────────────────────────────────────
    print("\n  ┌─ All four Bell states ──────────────────────────────────────────┐")
    for name in ("phi+", "phi-", "psi+", "psi-"):
        _, s, _ = QuantumAlgorithms.bell_state(name)
        print(Visualiser.ascii_bar(s, f"|{name.replace('phi','Φ').replace('psi','Ψ')}⟩"))


def demo_three_qubit():
    """Demonstrate 3-qubit systems: GHZ state and Toffoli gate."""
    section("DEMO 3 — Three Qubits: GHZ State and Toffoli Gate")

    print("""
  3 qubits → 8 basis states: |000⟩ through |111⟩.
  State vector has 8 complex amplitudes.
    """)

    # ── GHZ state ─────────────────────────────────────────────────────────────
    _, state, desc = QuantumAlgorithms.ghz_state(3)
    print(f"  {desc}")
    print(state)
    print(Visualiser.ascii_bar(state, "3-qubit GHZ state"))

    # ── Toffoli gate demo ──────────────────────────────────────────────────────
    section_sub = "  ┌─ Toffoli gate (CCNOT): flips target iff both controls = 1 ──────┐"
    print(section_sub)
    print("""
  Truth table:
    |control1, control2, target⟩ → |control1, control2, target XOR (c1 AND c2)⟩
    |000⟩ → |000⟩
    |001⟩ → |001⟩
    |010⟩ → |010⟩
    |011⟩ → |011⟩
    |100⟩ → |100⟩
    |101⟩ → |101⟩
    |110⟩ → |111⟩   ← both controls 1, target flipped 0→1
    |111⟩ → |110⟩   ← both controls 1, target flipped 1→0
    """)

    for init in [0b110, 0b111, 0b010]:
        q = QuantumState(3, init)
        Gates.apply_three(q, Gates.Toffoli(), 0, 1, 2)
        print(f"  Toffoli on |{init:03b}⟩ → ", end="")
        result = int(np.argmax(q.probabilities()))
        print(f"|{result:03b}⟩")


def demo_four_qubit():
    """Demonstrate the full 4-qubit system."""
    section("DEMO 4 — Four Qubits: Full 16-State Hilbert Space")

    print("""
  4 qubits → 16 basis states: |0000⟩ through |1111⟩.
  State vector has 16 complex amplitudes.

  H^⊗4|0000⟩ creates a UNIFORM superposition over all 16 states,
  each with amplitude 1/4 and probability 1/16 = 6.25%.
    """)

    qc = QuantumCircuit(4)
    for i in range(4):
        qc.h(i)
    state = qc.run()
    print(state)
    print(Visualiser.ascii_bar(state, "H⊗H⊗H⊗H |0000⟩ — uniform superposition (4 qubits)"))

    # ── 4-qubit GHZ ────────────────────────────────────────────────────────────
    print("\n  ┌─ 4-qubit GHZ state ────────────────────────────────────────────┐")
    _, state, desc = QuantumAlgorithms.ghz_state(4)
    print(f"  {desc}")
    print(state)
    print(Visualiser.ascii_bar(state, "4-qubit GHZ = (|0000⟩ + |1111⟩)/√2"))


def demo_deutsch_jozsa():
    """Run all three oracle variants of Deutsch-Jozsa."""
    section("DEMO 5 — Deutsch-Jozsa Algorithm")

    print("""
  Problem: is a black-box function f constant (same output for all inputs)
           or balanced (0 for half the inputs, 1 for the other half)?

  Classical worst case: 2^(n-1)+1 evaluations.
  Quantum: exactly 1 evaluation.

  If after ONE oracle call we measure all zeros → CONSTANT.
  Any non-zero result → BALANCED.
    """)

    for oracle in ("constant_0", "constant_1", "balanced"):
        qc, state, desc = QuantumAlgorithms.deutsch_jozsa(n_input=2, oracle_type=oracle)
        print(f"\n  Oracle: {oracle}")
        print(f"  {desc}")
        print(Visualiser.ascii_bar(state, f"Deutsch-Jozsa ({oracle})"))


def demo_grover():
    """Run Grover's search on 2-, 3-, and 4-qubit systems."""
    section("DEMO 6 — Grover's Quantum Search Algorithm")

    print("""
  Problem: find a marked item in an unstructured list of N items.
  Classical: O(N) queries.  Quantum: O(√N) queries.

  The algorithm amplifies the amplitude of the target state.
  After ~π√N/4 iterations, measuring gives the target with high probability.
    """)

    configs = [
        (2, 2,  "4  items, target |10⟩"),
        (2, 3,  "4  items, target |11⟩"),
        (3, 5,  "8  items, target |101⟩"),
        (4, 9,  "16 items, target |1001⟩"),
        (4, 14, "16 items, target |1110⟩"),
    ]

    for n, t, label in configs:
        qc, state, desc = QuantumAlgorithms.grover(n_qubits=n, target=t)
        print(f"\n  {label}")
        print(f"  {desc}")
        print(Visualiser.ascii_bar(state, f"Grover n={n} target={t}"))


def demo_teleportation():
    """Show the quantum teleportation protocol."""
    section("DEMO 7 — Quantum Teleportation")

    print("""
  Teleportation transfers an unknown quantum state |ψ⟩ from Alice to Bob
  using one pre-shared Bell pair + two classical bits.

  It does NOT allow FTL communication: the 2 classical bits must travel
  at light speed.  But the quantum information (the exact α, β of |ψ⟩)
  is transferred perfectly, without ever being measured directly.
    """)

    qc, state, desc = QuantumAlgorithms.teleportation()
    print(desc)
    print(state)
    print(Visualiser.ascii_bar(state, "Teleportation — joint state before Alice measures"))


def demo_phase_gates():
    """Illustrate how Z, S, T create distinct phase patterns in superposition."""
    section("DEMO 8 — Phase Gates: Z, S, T Acting on Superposition")

    print("""
  Phase gates leave |0⟩ unchanged but rotate the phase of |1⟩:
      Z: +π  (180°)  — multiplies |1⟩ amplitude by -1
      S: +π/2 (90°)  — multiplies |1⟩ amplitude by i
      T: +π/4 (45°)  — multiplies |1⟩ amplitude by e^(iπ/4)

  A phase change is invisible to direct measurement of a single qubit
  (|αe^(iφ)|² = |α|²) but creates INTERFERENCE when combined with
  other gates — this is how quantum algorithms get their power.
    """)

    for gate_fn, name, phase_label in [
        (Gates.Z, "Z", "+π"),
        (Gates.S, "S", "+π/2"),
        (Gates.T, "T", "+π/4"),
    ]:
        q = QuantumState(1)
        Gates.apply_single(q, Gates.H(), 0)    # create |+⟩
        Gates.apply_single(q, gate_fn(), 0)    # apply phase gate
        Gates.apply_single(q, Gates.H(), 0)    # interfere back
        print(f"  H → {name}({phase_label}) → H  on |0⟩:")
        print(q)
        print(Visualiser.ascii_bar(q, f"H {name} H |0⟩"))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║            FOUR-QUBIT QUANTUM COMPUTING SIMULATOR                          ║
║            A fully annotated educational Python / NumPy implementation     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Python  : """ + sys.version.split()[0] + """
  NumPy   : """ + np.__version__ + """
  Matplotlib: """ + ("available ✓" if MATPLOTLIB_AVAILABLE else "not installed — ASCII charts only") + """

  This simulation is EXACT: it uses the full 2ⁿ-dimensional state vector
  and represents every gate as a unitary matrix.  No approximations are made.
  Measurement is simulated stochastically using the Born rule.
""")

    np.random.seed(42)    # reproducible random measurements for the demo

    demo_single_qubit()
    demo_two_qubit()
    demo_three_qubit()
    demo_four_qubit()
    demo_deutsch_jozsa()
    demo_grover()
    demo_teleportation()
    demo_phase_gates()

    section("SUMMARY — WHAT WE COVERED")
    print("""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Concept             │ Key idea                                           │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Qubit               │ Complex 2-vector; |0⟩ and |1⟩ are basis states   │
  │ Superposition       │ H gate creates (|0⟩+|1⟩)/√2                      │
  │ Measurement         │ Born rule: P(i) = |αᵢ|²; collapses the state     │
  │ X / Y / Z gates     │ Bit-flip / Y-rotation / phase-flip                │
  │ S / T gates         │ π/2 and π/4 phase rotations                       │
  │ CNOT gate           │ Controlled flip; creates entanglement             │
  │ Bell states         │ Maximally entangled 2-qubit states                │
  │ GHZ states          │ Maximally entangled n-qubit states                │
  │ Toffoli gate        │ Quantum AND; universal for classical computation  │
  │ Deutsch-Jozsa       │ Exponential speedup via phase kick-back           │
  │ Grover's search     │ Quadratic speedup via amplitude amplification     │
  │ Teleportation       │ Transfer |ψ⟩ via entanglement + 2 classical bits  │
  └──────────────────────────────────────────────────────────────────────────┘

  All results above are EXACT quantum mechanical predictions.
  Run with matplotlib installed for graphical probability charts.
""")
