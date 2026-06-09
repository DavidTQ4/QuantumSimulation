# Four-Qubit Quantum Computing Simulator

A fully annotated, educational quantum computing simulator written in pure Python and NumPy. No quantum hardware required — this runs exact quantum mechanics on your laptop.

---

## Quick Start

```bash
pip install numpy matplotlib   # matplotlib is optional but gives graphical charts
python quantum_simulator.py
```

---

## Table of Contents

1. [The Core Idea — What a Quantum State Actually Is](#1-the-core-idea)
2. [QuantumState — The State Vector](#2-quantumstate)
3. [Gates — How We Transform the State](#3-gates)
4. [Gate Application — The Tensor Contraction Engine](#4-gate-application)
5. [QuantumCircuit — Composing Gates into Algorithms](#5-quantumcircuit)
6. [Measurement — The Born Rule and Collapse](#6-measurement)
7. [Visualiser — Seeing the Probabilities](#7-visualiser)
8. [Demo 1 — Single Qubit](#8-demo-1--single-qubit)
9. [Demo 2 — Two Qubits and Entanglement](#9-demo-2--two-qubits-and-entanglement)
10. [Demo 3 — Three Qubits and the Toffoli Gate](#10-demo-3--three-qubits-and-the-toffoli-gate)
11. [Demo 4 — Four Qubits: The Full System](#11-demo-4--four-qubits)
12. [Demo 5 — Deutsch-Jozsa Algorithm](#12-demo-5--deutsch-jozsa-algorithm)
13. [Demo 6 — Grover's Search Algorithm](#13-demo-6--grovers-search-algorithm)
14. [Demo 7 — Quantum Teleportation](#14-demo-7--quantum-teleportation)
15. [Demo 8 — Phase Gates and Interference](#15-demo-8--phase-gates-and-interference)
16. [Mathematical Reference](#16-mathematical-reference)

---

## 1. The Core Idea

Classical computers store information as **bits** — each one is definitely 0 or definitely 1.

A quantum computer stores information as **qubits**. A qubit is not just 0 or 1: before it is measured it exists in a **superposition** of both, described by two complex numbers called **amplitudes**.

The most general state of a single qubit is:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

where α and β are complex numbers satisfying:

```
|α|² + |β|² = 1     ← the normalisation condition
```

The vertical-bar-and-angle notation `|…⟩` is called **Dirac (ket) notation** — it is just a physicist's way of writing a vector. `|0⟩` and `|1⟩` are the two **basis vectors** (like the x̂ and ŷ unit vectors in 2D space, but for a 2D complex space).

When you **measure** the qubit, the superposition collapses:
- You get the outcome `0` with probability `|α|²`
- You get the outcome `1` with probability `|β|²`

After measurement the qubit is no longer in superposition — it is definitively in whichever state was observed.

### Scaling to n Qubits

The power of quantum computing comes from how the state space grows:

| Qubits | Basis states | Amplitudes needed |
|--------|-------------|-------------------|
| 1      | 2           | 2                 |
| 2      | 4           | 4                 |
| 3      | 8           | 8                 |
| 4      | 16          | 16                |
| n      | 2ⁿ          | 2ⁿ                |

A 4-qubit state is a vector of **16 complex numbers**. A quantum computer manipulates all 2ⁿ amplitudes simultaneously — this is the origin of quantum parallelism.

---

## 2. QuantumState

**File location:** `quantum_simulator.py` — class `QuantumState`

This class is the heart of the simulator. It holds the complete quantum state as a NumPy array of complex numbers.

### What it stores

```python
self.amplitudes = np.zeros(self.n_states, dtype=complex)
self.amplitudes[initial_state] = 1.0 + 0j
```

For a 3-qubit system initialised in `|000⟩`, this creates:

```
Index:      0       1       2       3       4       5       6       7
State:   |000⟩  |001⟩  |010⟩  |011⟩  |100⟩  |101⟩  |110⟩  |111⟩
Value:    1+0j   0+0j   0+0j   0+0j   0+0j   0+0j   0+0j   0+0j
```

### The index ↔ basis state mapping

The index `i` maps to the basis state whose binary representation is `i`, with **qubit 0 as the most significant bit**:

```
Index 5 = 0b101  →  |101⟩  →  qubit0=1, qubit1=0, qubit2=1
```

This convention (MSB first) is used consistently throughout the simulator so that the CNOT gate's "control" qubit is always qubit 0.

### Normalisation check

Every valid quantum state must satisfy the Born rule constraint. The `is_normalised()` method verifies:

```python
np.isclose(np.sum(np.abs(self.amplitudes)**2), 1.0)
```

The simulator checks this automatically; every gate being unitary guarantees it stays true.

---

## 3. Gates

**File location:** `quantum_simulator.py` — class `Gates`

A quantum gate is a **unitary matrix** `U` that transforms the state:

```
|ψ'⟩ = U|ψ⟩
```

Unitary means `U†U = I` (where `†` is the conjugate transpose). This guarantees:
1. Normalisation is preserved — gates never violate probability
2. The operation is **reversible** — `U⁻¹ = U†`

Here is every gate the simulator implements, with its matrix and what it does physically.

### Single-Qubit Gates

#### Identity — `I`
```
I = [ 1  0 ]
    [ 0  1 ]
```
Does nothing. Used as a placeholder when building multi-qubit operations from tensor products.

---

#### Pauli-X — `X` (the quantum NOT gate)
```
X = [ 0  1 ]
    [ 1  0 ]
```
**Action:**
```
X|0⟩ = |1⟩     (flips 0 to 1)
X|1⟩ = |0⟩     (flips 1 to 0)
```
On a superposition: `X(α|0⟩ + β|1⟩) = α|1⟩ + β|0⟩` — it swaps the two amplitudes.

This is the exact quantum analogue of a classical NOT gate.

---

#### Pauli-Y — `Y`
```
Y = [  0  -i ]
    [  i   0 ]
```
**Action:**
```
Y|0⟩ =  i|1⟩
Y|1⟩ = -i|0⟩
```
Combines a bit-flip with a phase change. Together with X and Z it forms the **Pauli group**, which generates all single-qubit unitaries.

---

#### Pauli-Z — `Z` (phase flip)
```
Z = [ 1   0 ]
    [ 0  -1 ]
```
**Action:**
```
Z|0⟩ =  |0⟩     (unchanged)
Z|1⟩ = -|1⟩     (sign flipped)
```
On a superposition: `Z(α|0⟩ + β|1⟩) = α|0⟩ - β|1⟩`

The probabilities `|α|²` and `|β|²` are identical before and after — the phase flip is **invisible to direct measurement** of that qubit alone. Its effect is only felt through **interference** with other gates, which is precisely how quantum algorithms encode information.

---

#### Hadamard — `H` (the superposition gate)
```
H = (1/√2) × [ 1   1 ]
              [ 1  -1 ]
```
**Action:**
```
H|0⟩ = (|0⟩ + |1⟩)/√2  =  |+⟩   (equal superposition, both amplitudes +1/√2)
H|1⟩ = (|0⟩ - |1⟩)/√2  =  |−⟩   (equal superposition, amplitude of |1⟩ is -1/√2)
```

This is the most important single-qubit gate. Starting from `|0⟩`, a single H creates a **50/50 superposition** — measuring this state gives 0 half the time and 1 half the time, completely at random.

Applied to all n qubits of an `|0…0⟩` state:

```
H⊗H⊗…⊗H |0…0⟩ = (1/√2ⁿ) Σᵢ |i⟩
```

This creates a **uniform superposition over all 2ⁿ basis states** simultaneously. A 4-qubit uniform superposition has all 16 amplitudes equal to 1/4, each with probability 1/16 = 6.25%.

H is its own inverse: `H² = I`, so applying it twice returns to the original state.

---

#### S gate — `S` (quarter-turn phase gate, also written √Z)
```
S = [ 1  0 ]
    [ 0  i ]
```
**Action:**
```
S|0⟩ = |0⟩
S|1⟩ = i|1⟩     (adds a phase of π/2 to |1⟩)
```

Applying S twice gives Z: `S² = Z`. Used in the quantum Fourier transform.

---

#### T gate — `T` (eighth-turn phase gate, also written ⁴√Z or π/8 gate)
```
T = [ 1         0        ]
    [ 0   e^(iπ/4)       ]
  = [ 1         0        ]
    [ 0   (1+i)/√2       ]
```
**Action:**
```
T|0⟩ = |0⟩
T|1⟩ = e^(iπ/4)|1⟩     (adds a phase of π/4 to |1⟩)
```

Applying T four times gives Z: `T⁴ = Z`; twice gives S: `T² = S`.

**Why T is special:** The set `{H, T}` is **universal** — any unitary transformation on any number of qubits can be approximated to arbitrary precision using only these two gates (Solovay–Kitaev theorem). T is the "non-Clifford" gate that provides the extra computational power beyond the Clifford group.

---

#### Rotation gates — `Rx(θ)`, `Ry(θ)`, `Rz(θ)`

These rotate the qubit state around the X, Y, or Z axis of the **Bloch sphere** by angle θ.

The Bloch sphere is a unit sphere where every point on the surface corresponds to a valid single-qubit state. The north pole is `|0⟩`, the south pole is `|1⟩`, and all superpositions lie on the equator and between.

```
Rx(θ) = [ cos(θ/2)    -i·sin(θ/2) ]
        [ -i·sin(θ/2)  cos(θ/2)   ]

Ry(θ) = [ cos(θ/2)   -sin(θ/2) ]
        [ sin(θ/2)    cos(θ/2)  ]

Rz(θ) = [ e^(-iθ/2)     0      ]
        [    0        e^(+iθ/2) ]
```

At `θ = π`: `Rx(π) ≈ -iX`, `Ry(π) ≈ -iY`, `Rz(π) ≈ -iZ` (equivalent up to global phase).

---

#### Phase gate — `Phase(φ)`
```
P(φ) = [ 1       0    ]
       [ 0   e^(iφ)   ]
```
The general phase gate. Special cases:
- `P(π)   = Z`
- `P(π/2) = S`
- `P(π/4) = T`

---

### Two-Qubit Gates

Two-qubit gates act on a pair of qubits and are represented as 4×4 matrices. The basis ordering is `|00⟩, |01⟩, |10⟩, |11⟩` with qubit 0 as the left (most significant) bit.

#### CNOT — `CNOT` (Controlled-NOT, the entangling gate)
```
CNOT = [ 1  0  0  0 ]   |00⟩ → |00⟩
       [ 0  1  0  0 ]   |01⟩ → |01⟩
       [ 0  0  0  1 ]   |10⟩ → |11⟩  ← control=1, target flipped
       [ 0  0  1  0 ]   |11⟩ → |10⟩  ← control=1, target flipped
```

**What it does:** If the control qubit (qubit 0) is `|1⟩`, flip the target qubit (qubit 1). If the control is `|0⟩`, do nothing.

**Why it creates entanglement:**

Start with `|+0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2`

After CNOT: `(|00⟩ + |11⟩)/√2`

This result **cannot** be written as `(some state for qubit 0) ⊗ (some state for qubit 1)`. The two qubits are now **entangled** — their fates are linked. Measuring qubit 0 and finding `0` instantly tells you qubit 1 is also `0`; finding `1` tells you qubit 1 is also `1`.

---

#### CZ — `CZ` (Controlled-Z)
```
CZ = [ 1  0  0   0 ]   |00⟩ →  |00⟩
     [ 0  1  0   0 ]   |01⟩ →  |01⟩
     [ 0  0  1   0 ]   |10⟩ →  |10⟩
     [ 0  0  0  -1 ]   |11⟩ → -|11⟩  ← phase flip only when both = 1
```

Unlike CNOT, CZ is **symmetric** — both qubits play equal roles. It adds a `-1` phase only when both qubits are in state `|1⟩`.

---

#### SWAP — `SWAP`
```
SWAP = [ 1  0  0  0 ]   |00⟩ → |00⟩
       [ 0  0  1  0 ]   |01⟩ → |10⟩  ← swapped
       [ 0  1  0  0 ]   |10⟩ → |01⟩  ← swapped
       [ 0  0  0  1 ]   |11⟩ → |11⟩
```

Exchanges the complete quantum states of two qubits. Can be decomposed into three CNOT gates: `SWAP = CNOT(0→1) · CNOT(1→0) · CNOT(0→1)`.

---

### Three-Qubit Gates

#### Toffoli — `Toffoli` (CCNOT, Controlled-Controlled-NOT)

An 8×8 matrix acting on 3 qubits. It flips the **target** qubit (qubit 2) only when **both** control qubits (qubits 0 and 1) are `|1⟩`.

```
|000⟩ → |000⟩     |100⟩ → |100⟩
|001⟩ → |001⟩     |101⟩ → |101⟩
|010⟩ → |010⟩     |110⟩ → |111⟩  ← both controls 1, target flipped
|011⟩ → |011⟩     |111⟩ → |110⟩  ← both controls 1, target flipped
```

This is the quantum analogue of the classical **AND gate** (in reversible form). The Toffoli gate is **universal for classical reversible computation**: any classical circuit can be built from Toffoli gates alone.

---

## 4. Gate Application

**File location:** `quantum_simulator.py` — `Gates.apply_single`, `Gates.apply_two`, `Gates.apply_three`

This is where the simulation actually does its work.

### The Naive Approach (and why we don't use it)

When a single-qubit gate `U` acts on qubit `k` of an n-qubit system, the full operation on the state vector is the **tensor product**:

```
U_full = I ⊗ I ⊗ … ⊗ U ⊗ … ⊗ I    (U at position k)
```

For 4 qubits this would be a 16×16 matrix — but for 30 qubits it would be 10⁹ × 10⁹. Materialising this matrix is completely impractical.

### The Tensor Contraction Approach (what we actually do)

Instead, the simulator treats the 1D amplitude array as a **multi-dimensional tensor** by reshaping it:

```python
tensor = state.amplitudes.reshape([2] * n)   # shape: (2, 2, 2, ..., 2)
```

Now each axis of the tensor corresponds to one qubit. Applying a 2×2 gate matrix to qubit `k` is a **tensor contraction** along axis `k`:

```python
result = np.tensordot(gate, tensor, axes=([1], [qubit]))
result = np.moveaxis(result, 0, qubit)
state.amplitudes = result.reshape(state.n_states)
```

This is mathematically identical to the full matrix multiplication but:
- Uses O(2ⁿ) memory, not O(4ⁿ)
- Runs in O(2ⁿ) time, not O(4ⁿ)

The two-qubit version contracts over two axes simultaneously; the three-qubit version over three axes. The axis-rearrangement with `moveaxis` / `transpose` is needed to put the output axes back in the correct qubit positions after the contraction moves them to the front.

---

## 5. QuantumCircuit

**File location:** `quantum_simulator.py` — class `QuantumCircuit`

`QuantumCircuit` is a reusable, ordered list of gate operations. It does not hold any quantum state itself — it is purely a description of what to do.

### How it works

```python
qc = QuantumCircuit(2)    # 2-qubit circuit
qc.h(0)                   # add H gate on qubit 0 to the list
qc.cnot(0, 1)             # add CNOT(control=0, target=1) to the list
state = qc.run()          # now create a fresh |00⟩ and apply gates in order
```

Internally each gate is stored as a tuple `(label, matrix, [qubit_indices])`. When `run()` is called:

1. A fresh `QuantumState` is created in `|0…0⟩`
2. Each gate is applied in order using the tensor contraction engine
3. The final `QuantumState` is returned

Because `run()` always starts from a fresh state, the same circuit can be called multiple times with no side effects.

---

## 6. Measurement

**File location:** `quantum_simulator.py` — `QuantumState.measure`, `QuantumState.measure_qubit`

### Full measurement — `measure(shots=1)`

Measurement is the **irreversible** process by which a quantum state collapses into one of the classical basis states.

**Before measurement:** `|ψ⟩ = Σ αᵢ|i⟩` — all basis states coexist in superposition.

**After measurement:** the system is found in state `|k⟩` with probability `P(k) = |αₖ|²`.

The implementation uses NumPy's weighted random sampling, which faithfully implements the Born rule:

```python
probs = np.abs(self.amplitudes) ** 2
outcome = int(np.random.choice(self.n_states, p=probs))
```

After sampling, the state **collapses**:

```python
self.amplitudes = np.zeros(self.n_states, dtype=complex)
self.amplitudes[outcome] = 1.0 + 0j
```

Any subsequent measurement of the same (now collapsed) state always returns the same answer.

### Partial measurement — `measure_qubit(qubit)`

Measures a **single qubit** without touching the rest of the system.

1. **Compute marginal probability** — sum `|αᵢ|²` over all basis states where qubit `k` = 1 to get `P(qubit=1)`; the rest is `P(qubit=0)`.
2. **Sample** outcome 0 or 1 using those probabilities.
3. **Project** — zero out all amplitudes inconsistent with the outcome.
4. **Renormalise** — divide surviving amplitudes by their norm so the state is valid again.

This partial collapse is what happens in real quantum hardware when you read one qubit of a multi-qubit register, and is used in the entanglement correlation demo and the teleportation protocol.

---

## 7. Visualiser

**File location:** `quantum_simulator.py` — class `Visualiser`

### ASCII bar chart — `Visualiser.ascii_bar(state, title)`

Prints a probability histogram to the terminal. For each basis state it shows:
- The ket label (e.g. `|101⟩`)
- A filled bar proportional to the probability
- The probability as a percentage

Example output for the 2-qubit Bell state `|Φ+⟩ = (|00⟩+|11⟩)/√2`:
```
  |00⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
  |01⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |10⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
  |11⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
```

### Graphical chart — `Visualiser.plot(state, title)`

If matplotlib is installed, produces a two-panel figure:
- **Left panel:** probability bar chart with colour-coded bars and value labels
- **Right panel:** Argand diagram showing each amplitude as a point in the complex plane, with the unit circle for reference

The Argand diagram reveals phase information that the bar chart hides — two states with the same probability `|α|²` can have very different phases `arg(α)`, and those phases determine how the state behaves under further gates.

---

## 8. Demo 1 — Single Qubit

**Function:** `demo_single_qubit()`

This demo walks through every concept for the simplest possible system.

### Starting state `|0⟩`
```
amplitudes = [1+0j, 0+0j]
```
Probability 100% of measuring 0. This is the equivalent of a classical bit set to 0.

### After X gate → `|1⟩`
```
amplitudes = [0+0j, 1+0j]
```
The X matrix swaps the two amplitudes. Probability 100% of measuring 1.

### After H gate → `|+⟩`
```
amplitudes = [0.707+0j, 0.707+0j]   (1/√2 ≈ 0.707)
```
Both amplitudes are equal. Measuring gives 0 or 1 with equal probability — a truly random outcome. Running this 20 times produces approximately 10 zeros and 10 ones (the demo shows the actual random results).

### H applied twice → back to `|0⟩`
Applying H twice is the identity: `H² = I`. The second H causes the amplitudes to **interfere** — the positive and negative terms cancel back to the original state.

### After Z on `|+⟩` → `|−⟩`
```
amplitudes = [0.707+0j, -0.707+0j]
```
The probabilities are **identical** to `|+⟩` — both states give 50/50 on measurement. The difference is the sign of the `|1⟩` amplitude. This phase difference only becomes observable when further gates are applied (see Demo 8).

### S and T gates on `|+⟩`
S multiplies the `|1⟩` amplitude by `i`; T multiplies it by `e^(iπ/4) = (1+i)/√2`. Again the **probabilities are unchanged** but the complex phase of the amplitude shifts. These phases accumulate through circuits and determine interference outcomes.

---

## 9. Demo 2 — Two Qubits and Entanglement

**Function:** `demo_two_qubit()`

### Product state: H on qubit 0 only
```
|ψ⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
```
```
  |00⟩  ██████████████████████░░░░░░░░░░░░░░   50.00%
  |01⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |10⟩  ██████████████████████░░░░░░░░░░░░░░   50.00%
  |11⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
```
This is a **product state** — qubit 0 is in superposition but qubit 1 is definitely `|0⟩`. Measuring qubit 1 always gives 0, regardless of what qubit 0 does.

### Bell state `|Φ+⟩`: H then CNOT

**Circuit:**
```
q0: ─ H ─●─
          │
q1: ─────X─
```

**Step by step:**
1. Start: `|00⟩`
2. After H on q0: `(|00⟩ + |10⟩)/√2`
3. After CNOT: `(|00⟩ + |11⟩)/√2`

```
  |00⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
  |01⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |10⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |11⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
```

Notice: `|01⟩` and `|10⟩` have **zero probability**. This state cannot be written as a product state — the qubits are **entangled**.

### Entanglement correlation demo

The demo prepares `|Φ+⟩` ten times and measures each qubit separately. Every trial the two outcomes match perfectly (both 0 or both 1), demonstrating the quantum correlation.

### All four Bell states

The four Bell states are the **maximally entangled** 2-qubit states. They form an orthonormal basis for the 4-dimensional 2-qubit Hilbert space.

| State | Formula | Circuit |
|-------|---------|---------|
| `\|Φ+⟩` | `(\|00⟩ + \|11⟩)/√2` | H(q0), CNOT |
| `\|Φ-⟩` | `(\|00⟩ - \|11⟩)/√2` | H(q0), Z(q0), CNOT |
| `\|Ψ+⟩` | `(\|01⟩ + \|10⟩)/√2` | X(q1), H(q0), CNOT |
| `\|Ψ-⟩` | `(\|01⟩ - \|10⟩)/√2` | X(q1), H(q0), Z(q0), CNOT |

---

## 10. Demo 3 — Three Qubits and the Toffoli Gate

**Function:** `demo_three_qubit()`

### GHZ state (3-qubit)

The **Greenberger-Horne-Zeilinger** state is:

```
|GHZ⟩ = (|000⟩ + |111⟩)/√2
```

**Circuit:**
```
q0: ─ H ─●────
          │
q1: ─────X─●──
            │
q2: ─────────X─
```

Only `|000⟩` and `|111⟩` have non-zero probability (each 50%). All three qubits are entangled — measuring any one determines all three. The GHZ state has no classical analogue and has been used to test the foundations of quantum mechanics.

### Toffoli gate — classical AND in quantum form

The demo applies the Toffoli gate to several basis states to show its truth table:

```
Toffoli on |110⟩ → |111⟩   (both controls 1, target 0 → flipped to 1)
Toffoli on |111⟩ → |110⟩   (both controls 1, target 1 → flipped to 0)
Toffoli on |010⟩ → |010⟩   (only one control 1, target unchanged)
```

---

## 11. Demo 4 — Four Qubits

**Function:** `demo_four_qubit()`

### Uniform superposition: H⊗H⊗H⊗H

Applying H to all 4 qubits simultaneously:

```
H⊗H⊗H⊗H |0000⟩ = (1/4) Σ|i⟩   for i = 0..15
```

All 16 amplitudes equal `1/4`; all 16 probabilities equal `1/16 = 6.25%`. The state encodes all 16 possible 4-bit numbers simultaneously.

### 4-qubit GHZ

```
|GHZ₄⟩ = (|0000⟩ + |1111⟩)/√2
```

Of the 16 possible states, only two have non-zero probability. All four qubits are perfectly correlated.

---

## 12. Demo 5 — Deutsch-Jozsa Algorithm

**Function:** `demo_deutsch_jozsa()`

### The Problem

You are given a black-box function `f: {0,1}ⁿ → {0,1}` that is **guaranteed** to be either:
- **Constant** — `f(x)` returns the same value (all 0 or all 1) for every input
- **Balanced** — `f(x) = 0` for exactly half the inputs and `f(x) = 1` for the other half

Determine which case applies.

**Classical worst case:** You might need to query up to `2^(n-1) + 1` inputs before you can be certain.

**Quantum:** You need exactly **one** query. This is an exponential speedup.

### The Circuit (n=2 input qubits + 1 ancilla)

```
q0: ─ H ────[Oracle]── H ──M
q1: ─ H ────[Oracle]── H ──M
q2: ─ X ─ H ─[Oracle]──────
```

### Step-by-step explanation

**Step 1 — Setup:**
- Input qubits (q0, q1) initialised to `|0⟩`
- Ancilla qubit (q2) initialised to `|1⟩` via an X gate

**Step 2 — H on all qubits:**
- Input qubits enter a uniform superposition over all 4 inputs simultaneously
- Ancilla becomes `|−⟩ = (|0⟩ - |1⟩)/√2`

**Step 3 — Oracle query (phase kick-back trick):**

The oracle computes `f(x)` but, because the ancilla is in `|−⟩`, it **writes the result as a phase** rather than flipping a bit:

```
Uf|x⟩|−⟩ = (-1)^f(x) |x⟩|−⟩
```

The ancilla is unchanged. The input register now encodes `f(x)` as a `±1` phase on each component of the superposition — simultaneously for **all** inputs.

**Step 4 — H on input qubits:**

The H gates cause interference. The phases encode whether `f` is constant or balanced:
- **Constant** → all phases equal → constructive interference at `|00⟩` → measuring input gives `|00⟩` with certainty
- **Balanced** → phases cancel at `|00⟩` → measuring input gives anything **except** `|00⟩`

**Step 5 — Measure:**
- All zeros in the input register → **CONSTANT**
- Any non-zero → **BALANCED**

### What the simulator shows

Three runs with different oracles:

| Oracle type | Expected result | Probability of `\|00⟩` in input register |
|-------------|----------------|------------------------------------------|
| `constant_0` | CONSTANT | 100% |
| `constant_1` | CONSTANT | 100% |
| `balanced`   | BALANCED | 0%   |

---

## 13. Demo 6 — Grover's Search Algorithm

**Function:** `demo_grover()`

### The Problem

Search an **unstructured** list of `N = 2ⁿ` items to find the one that satisfies a predicate (the "marked" item).

**Classical:** On average `N/2` queries; worst case `N`.

**Quantum:** `O(√N)` queries — a **quadratic speedup**. For 4 qubits (N=16) this is 4 queries instead of 8 on average.

### The Geometric Intuition

Imagine the state vector as a pointer in a 2D plane. The two relevant directions are:
- `|t⟩` — the target state
- `|s⊥⟩` — the superposition of everything except the target

Initially the pointer is nearly aligned with `|s⊥⟩` (close to equal superposition). Each iteration rotates the pointer by a fixed angle `2θ` toward `|t⟩`, where `sin(θ) = 1/√N`.

After approximately `π/(4θ) ≈ π√N/4` rotations, the pointer is nearly aligned with `|t⟩`. Measuring then finds the target with high probability.

### The Circuit (one iteration shown)

```
q0: ─ H ──[Oracle]─ H ─ X ─────●──── X ─ H ─
q1: ─ H ──[Oracle]─ H ─ X ─[phase]── X ─ H ─
```

Each iteration consists of two operations:

**1. Oracle Oₜ (phase flip of target)**

Flips the phase of the marked state `|t⟩` and leaves all others alone:
```
|x⟩ → (-1)^δ(x,t) |x⟩
```

Implementation: X gates to flip the qubits where the target has a `0` bit, then a multi-controlled Z (CZ for n=2, Toffoli-based for n=3,4), then X gates to undo the flips.

**2. Diffusion operator D (inversion about the mean)**

```
D = 2|s⟩⟨s| - I
```

where `|s⟩` is the uniform superposition. This reflects all amplitudes about their mean value, amplifying the marked state (whose amplitude is now negative from the oracle) and suppressing all others.

Implementation: `H^⊗n → X^⊗n → multi-controlled-Z → X^⊗n → H^⊗n`

### Simulator results

| System | Target | Iterations | P(target) after algorithm |
|--------|--------|-----------|--------------------------|
| 2 qubits (N=4)  | `\|10⟩` | 1 | ~100% |
| 2 qubits (N=4)  | `\|11⟩` | 1 | ~100% |
| 3 qubits (N=8)  | `\|101⟩` | 2 | ~97% |
| 4 qubits (N=16) | `\|1001⟩` | 3 | ~96% |

The probability is not exactly 100% because the rotation slightly overshoots the target direction after the optimal number of iterations. Running more iterations would actually decrease the probability again.

---

## 14. Demo 7 — Quantum Teleportation

**Function:** `demo_teleportation()`

### The Protocol

Teleportation transfers an **unknown quantum state** `|ψ⟩ = α|0⟩ + β|1⟩` from Alice to Bob using:
- 1 pre-shared **Bell pair** (entangled pair of qubits)
- 2 **classical bits** of communication

The state is transferred perfectly without Alice ever knowing α or β.

### The Three Qubits

- **q0** (Alice) — the **message** qubit in unknown state `|ψ⟩`. Here we use `Ry(π/3)|0⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩`
- **q1** (Alice) — Alice's half of the shared Bell pair
- **q2** (Bob) — Bob's half of the shared Bell pair

### Circuit

```
q0: ─ Ry(π/3) ──────────────●── H ── [measure → bit b₀]
                              │
q1: ─────────── H ──●────────X ────── [measure → bit b₁]
                     │
q2: ─────────────────X ────────────── [apply X if b₁=1, Z if b₀=1] → |ψ⟩
```

### Step by step

1. **Prepare message:** `Ry(π/3)` rotates q0 to the desired state `|ψ⟩`
2. **Create Bell pair:** H(q1) then CNOT(q1→q2) entangles q1 and q2
3. **Alice's Bell measurement:** CNOT(q0→q1) then H(q0) rotates into the Bell basis
4. **Classical communication:** Alice measures q0 and q1, gets 2 classical bits `b₀, b₁`
5. **Bob's correction:** Bob applies X to q2 if `b₁ = 1`, then Z if `b₀ = 1`

After the corrections, q2 is **exactly** in state `|ψ⟩`.

### Key points

- Alice's measurement collapses q0 and q1 — she no longer has the original `|ψ⟩`
- Bob cannot do anything useful with q2 until he receives the 2 classical bits
- This means teleportation does **not** allow faster-than-light communication
- The quantum information (the values of α and β) is transferred exactly, even though it was never measured or known

---

## 15. Demo 8 — Phase Gates and Interference

**Function:** `demo_phase_gates()`

This demo makes the most subtle point in quantum computing tangible: **phases are invisible to direct measurement but determine everything through interference**.

### The experiment

For each phase gate (Z, S, T), the circuit is:

```
|0⟩ ─ H ─ [phase gate] ─ H ─ measure
```

**Z gate (π phase):**
- After H: `(|0⟩ + |1⟩)/√2`
- After Z: `(|0⟩ - |1⟩)/√2`  ← probabilities unchanged, phase flipped
- After second H: these interfere destructively at `|0⟩` and constructively at `|1⟩`
- Result: `|1⟩` with 100% probability

**S gate (π/2 phase):**
- After S on `|+⟩`: `(|0⟩ + i|1⟩)/√2` ← same probabilities, i phase on `|1⟩`
- After second H: partial interference
- Result: 50% `|0⟩`, 50% `|1⟩`

**T gate (π/4 phase):**
- After T on `|+⟩`: `(|0⟩ + e^(iπ/4)|1⟩)/√2` ← phase of 45°
- After second H: partial constructive interference at `|0⟩`
- Result: 85.4% `|0⟩`, 14.6% `|1⟩`

The key insight: the **same input state** with **different phases** produces **completely different measurement outcomes** after a further gate. This is how quantum algorithms encode and process information — they engineer interference patterns so the right answers reinforce and the wrong answers cancel.

---

## 16. Mathematical Reference

### Dirac (Ket) Notation

| Symbol | Meaning |
|--------|---------|
| `\|ψ⟩` | A quantum state vector (ket) |
| `⟨ψ\|` | The conjugate transpose (bra) |
| `⟨φ\|ψ⟩` | Inner product (overlap between two states) |
| `\|ψ⟩⊗\|φ⟩` or `\|ψφ⟩` | Tensor product (composite system) |
| `\|0⟩, \|1⟩` | Computational basis states |

### Computational Basis

For n qubits, the computational basis states are:

```
|0…00⟩, |0…01⟩, |0…10⟩, …, |1…11⟩
```

The integer index `i` (0 to 2ⁿ-1) uniquely identifies each basis state via its binary representation, with qubit 0 as the most significant bit.

### Born Rule

The probability of measuring outcome `i` is:

```
P(i) = |αᵢ|²
```

where `αᵢ` is the complex amplitude of basis state `|i⟩`. All probabilities sum to 1.

### Unitarity

Every quantum gate `U` satisfies:

```
U†U = UU† = I
```

This is the quantum constraint equivalent to "operations must be reversible and probability-preserving".

### Tensor Product

For two systems with state vectors `|ψ⟩ ∈ ℂᵐ` and `|φ⟩ ∈ ℂⁿ`, the combined system lives in ℂᵐⁿ:

```
|ψ⟩ ⊗ |φ⟩  has length m×n
```

For the computational basis: `|ab⟩ = |a⟩ ⊗ |b⟩`, where the index of `|ab⟩` is `2a + b`.

---

## File Structure

```
QuantumSimulation/
├── quantum_simulator.py    # The complete simulator (~1700 lines)
└── requirements.txt        # numpy, matplotlib
```

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Yes | Complex vectors, matrix operations, random sampling |
| `matplotlib` | No | Graphical probability and Argand charts |

Install with:
```bash
pip install -r requirements.txt
```

The simulator runs fully with only NumPy (ASCII charts). Install matplotlib for the graphical Argand diagram view.
