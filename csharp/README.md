# Four-Qubit Quantum Computing Simulator — C# / .NET 8

A fully annotated, educational quantum computing simulator written in C# with no external dependencies beyond the .NET 8 runtime. Runs exact quantum mechanics on your laptop — no quantum hardware required.

---

## Quick Start

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/DavidTQ4/QuantumSimulation.git
cd QuantumSimulation/csharp

# Run the simulator
dotnet run --project QuantumSimulator
```

That's it. No NuGet packages. No external libraries. Just .NET 8.

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| .NET SDK    | 8.0+    | [Download from dot.net](https://dotnet.microsoft.com/download) |

Check your installed version:
```bash
dotnet --version
# Should print 8.x.x or higher
```

Install on Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0
```

Install on macOS (with Homebrew):
```bash
brew install dotnet
```

On Windows, download the installer from [https://dotnet.microsoft.com/download](https://dotnet.microsoft.com/download).

---

## Building and Running

### Run directly (build + run in one step)
```bash
dotnet run --project csharp/QuantumSimulator
```

### Build only (produces a binary in bin/)
```bash
dotnet build csharp/QuantumSimulator
```

### Publish a self-contained executable (no .NET runtime required on target machine)
```bash
dotnet publish csharp/QuantumSimulator -c Release -r linux-x64 --self-contained
# Output: csharp/QuantumSimulator/bin/Release/net8.0/linux-x64/publish/QuantumSimulator
```

Replace `linux-x64` with `win-x64` or `osx-arm64` for other platforms.

### Run from the compiled binary
```bash
./csharp/QuantumSimulator/bin/Release/net8.0/linux-x64/publish/QuantumSimulator
```

---

## Terminal Encoding (Windows)

The output uses Unicode characters (ket notation `|0⟩`, block bars `█`, tick marks `✓`).  
On Windows, set the terminal to UTF-8 before running:

```powershell
# PowerShell or Command Prompt
chcp 65001
dotnet run --project QuantumSimulator
```

Or set it permanently in Windows Terminal: **Settings → Profiles → Advanced → Text → Encoding → UTF-8**.

---

## Project Structure

```
csharp/
├── README.md                       ← this file
└── QuantumSimulator/
    ├── QuantumSimulator.csproj     ← project file (.NET 8, no NuGet deps)
    ├── QuantumState.cs             ← state vector, measurement, display
    ├── Gates.cs                    ← all gate matrices + applicators
    ├── QuantumCircuit.cs           ← composable gate pipeline
    ├── Visualiser.cs               ← ASCII probability bar chart
    ├── Algorithms.cs               ← Bell, GHZ, Deutsch-Jozsa, Grover, Teleportation
    └── Program.cs                  ← entry point — 8 annotated demos
```

---

## Table of Contents

1. [The Core Idea — What a Quantum State Actually Is](#1-the-core-idea)
2. [QuantumState.cs — The State Vector](#2-quantumstatecs)
3. [Gates.cs — How We Transform the State](#3-gatescs)
4. [Gate Application — The Pair-Iteration Engine](#4-gate-application)
5. [QuantumCircuit.cs — Composing Gates into Algorithms](#5-quantumcircuitcs)
6. [Measurement — The Born Rule and Collapse](#6-measurement)
7. [Visualiser.cs — Seeing the Probabilities](#7-visualisercs)
8. [Demo 1 — Single Qubit](#8-demo-1--single-qubit)
9. [Demo 2 — Two Qubits and Entanglement](#9-demo-2--two-qubits-and-entanglement)
10. [Demo 3 — Three Qubits and the Toffoli Gate](#10-demo-3--three-qubits-and-the-toffoli-gate)
11. [Demo 4 — Four Qubits: The Full System](#11-demo-4--four-qubits)
12. [Demo 5 — Deutsch-Jozsa Algorithm](#12-demo-5--deutsch-jozsa-algorithm)
13. [Demo 6 — Grover's Search Algorithm](#13-demo-6--grovers-search-algorithm)
14. [Demo 7 — Quantum Teleportation](#14-demo-7--quantum-teleportation)
15. [Demo 8 — Phase Gates and Interference](#15-demo-8--phase-gates-and-interference)
16. [Key C# vs Python Differences](#16-key-c-vs-python-differences)
17. [Mathematical Reference](#17-mathematical-reference)

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

The vertical-bar-and-angle notation `|…⟩` is **Dirac (ket) notation** — it is just a physicist's way of writing a vector. `|0⟩` and `|1⟩` are the two **basis vectors**.

When you **measure** the qubit, the superposition collapses:
- You get outcome `0` with probability `|α|²`
- You get outcome `1` with probability `|β|²`

### Scaling to n Qubits

| Qubits | Basis states | Amplitudes |
|--------|-------------|------------|
| 1      | 2           | 2          |
| 2      | 4           | 4          |
| 3      | 8           | 8          |
| 4      | 16          | 16         |
| n      | 2ⁿ          | 2ⁿ         |

A 4-qubit state is a vector of **16 complex numbers**. This simulator stores and manipulates all of them exactly.

---

## 2. QuantumState.cs

**Class:** `QuantumState`

The heart of the simulator. It stores the complete quantum state as a `Complex[]` array from `System.Numerics`.

### What it stores

```csharp
public Complex[] Amplitudes { get; private set; }
public int NQubits { get; }
public int NStates  { get; }   // = 1 << NQubits  (i.e. 2^NQubits)
```

Initialised in the constructor:

```csharp
Amplitudes = new Complex[NStates];
Amplitudes[initialState] = Complex.One;   // pure basis state, no superposition
```

For a 2-qubit system starting in `|00⟩`:

```
Index:  0       1       2       3
State: |00⟩   |01⟩   |10⟩   |11⟩
Value: 1+0i   0+0i   0+0i   0+0i
```

### The index ↔ basis state mapping

Index `i` maps to the basis state whose binary representation is `i`, with **qubit 0 as the most significant bit**:

```
Index 5 = 0b101  →  |101⟩  →  qubit0=1, qubit1=0, qubit2=1
```

In C#, the label is produced by:
```csharp
public string StateLabel(int index) =>
    $"|{Convert.ToString(index, 2).PadLeft(NQubits, '0')}⟩";
```

`Convert.ToString(index, 2)` converts to binary string; `PadLeft` zero-pads it to the correct width.

### Probabilities

```csharp
public double[] Probabilities()
{
    var probs = new double[NStates];
    for (int i = 0; i < NStates; i++)
        probs[i] = Amplitudes[i].Real * Amplitudes[i].Real
                 + Amplitudes[i].Imaginary * Amplitudes[i].Imaginary;
    return probs;
}
```

This computes `|αᵢ|²` for each amplitude. The `MagnitudeSquared()` extension method (`z.Real² + z.Imaginary²`) avoids the square root that `Complex.Magnitude` would compute — we only need the squared value.

### Creating a QuantumState

```csharp
// Start in |0⟩ (default)
var q = new QuantumState(1);

// Start in |11⟩  (index 3 in a 2-qubit system)
var q2 = new QuantumState(2, initialState: 3);

// Build from an existing amplitude array
var qs = QuantumState.FromAmplitudes(new Complex[] {
    new(1/Math.Sqrt(2), 0),   // |0⟩ amplitude
    new(1/Math.Sqrt(2), 0),   // |1⟩ amplitude
});
```

---

## 3. Gates.cs

**Static class:** `Gates`  
**Struct:** `GateMatrix`

A quantum gate is a **unitary matrix** `U` that transforms the state: `|ψ'⟩ = U|ψ⟩`

Unitary means `U†U = I`. This guarantees normalisation is preserved and the operation is reversible.

### GateMatrix struct

```csharp
public readonly struct GateMatrix
{
    public readonly Complex[] Data;   // row-major, length = Dim * Dim
    public readonly int Dim;          // 2^Arity
    public readonly int Arity;        // number of qubits the gate acts on

    public Complex this[int row, int col] => Data[row * Dim + col];
}
```

The `this[row, col]` indexer makes matrix element access read naturally. Internally the data is a flat `Complex[]` in row-major order, which is cache-friendly for the applicator loops.

### Single-qubit gates

Every gate is a static factory method returning a `GateMatrix`:

```csharp
Gates.I()           // Identity
Gates.X()           // Pauli-X / NOT
Gates.Y()           // Pauli-Y
Gates.Z()           // Pauli-Z / phase flip
Gates.H()           // Hadamard — creates superposition
Gates.S()           // S gate  — π/2 phase
Gates.T()           // T gate  — π/4 phase
Gates.Rx(theta)     // X-axis rotation
Gates.Ry(theta)     // Y-axis rotation
Gates.Rz(theta)     // Z-axis rotation
Gates.Phase(phi)    // General phase gate
```

Example — Hadamard:
```csharp
private const double Sqrt2Inv = 0.7071067811865476;  // 1/√2

public static GateMatrix H() => new(new Complex[]
{
    C(Sqrt2Inv),  C(Sqrt2Inv),
    C(Sqrt2Inv),  C(-Sqrt2Inv)
}, arity: 1);
```

The `C(re, im)` helper is just `new Complex(re, im)` — kept short to make matrices readable.

### Two-qubit gates

```csharp
Gates.CNOT()        // Controlled-NOT (the entangling gate)
Gates.CZ()          // Controlled-Z
Gates.SWAP()        // Swap two qubits
```

### Three-qubit gates

```csharp
Gates.Toffoli()     // CCNOT — quantum AND
```

---

## 4. Gate Application

**Methods:** `Gates.ApplySingle`, `Gates.ApplyTwo`, `Gates.ApplyThree`

This is where the simulation does its work. The key challenge: applying a small gate to one or two qubits of an n-qubit system without building the full `2ⁿ × 2ⁿ` matrix (which is impractical for large n).

### The pair-iteration approach

For a single-qubit gate on qubit `k` of an n-qubit system, the full unitary would be:

```
U_full = I ⊗ … ⊗ U ⊗ … ⊗ I    (U at position k)
```

Instead, the simulator observes that this operation **only mixes pairs of basis states that differ in bit position `k`**. So we iterate directly over those pairs.

**Bit position** for qubit `k` in an n-qubit system (MSB-first):
```csharp
int bitPos = n - 1 - k;
int stride = 1 << bitPos;   // distance between the two partners in the flat array
```

**For each basis state `i` where qubit `k` = 0**, the partner state (qubit `k` = 1) is `i | stride`. We apply the 2×2 gate to this pair:

```csharp
for (int outer = 0; outer < nStates; outer++)
{
    if (((outer >> bitPos) & 1) != 0) continue;   // skip the |…1…⟩ partner

    int i0 = outer;
    int i1 = outer | stride;

    Complex a0 = state.Amplitudes[i0];
    Complex a1 = state.Amplitudes[i1];

    state.Amplitudes[i0] = gate[0,0]*a0 + gate[0,1]*a1;
    state.Amplitudes[i1] = gate[1,0]*a0 + gate[1,1]*a1;
}
```

This is O(2ⁿ) in both time and memory — half the amplitude pairs are processed, each pair requiring 4 multiplications and 2 additions.

**ApplyTwo** extends the same idea to groups of four basis states that differ only in the two target qubits. For each `outer` index where both target qubits are 0, the four partners are:

```csharp
int i00 = outer;
int i01 = outer | stride1;
int i10 = outer | stride0;
int i11 = outer | stride0 | stride1;
```

The 4×4 matrix multiplication is applied to those four amplitudes. Each group of four is processed in isolation — the loops never double-process a group.

**ApplyThree** uses `stackalloc int[8]` for the eight partner indices and `Complex[8]` for the old amplitudes, then applies the 8×8 Toffoli matrix. `stackalloc` keeps the temporary buffer on the stack rather than the heap, avoiding per-group allocations in the inner loop.

### Why this is correct

The partition of basis states into non-overlapping groups (pairs for 1-qubit, quartets for 2-qubit, octets for 3-qubit) is exact. Within each group, the old amplitudes are read before any new ones are written (using temp variables `a0, a1` or the `old[]` buffer), so there is no ordering dependency between reads and writes.

---

## 5. QuantumCircuit.cs

**Class:** `QuantumCircuit`

A reusable, ordered list of gate operations. It holds no quantum state itself.

### Internal representation

Each gate is stored as a `file sealed record GateOp`:

```csharp
file sealed record GateOp(string Label, GateMatrix Matrix, int[] Qubits);
```

The `file` modifier (C# 11+) makes this type invisible outside `QuantumCircuit.cs`, keeping the public API clean.

### Building a circuit

```csharp
var qc = new QuantumCircuit(nQubits: 2);
qc.H(0);           // Hadamard on qubit 0
qc.CNOT(0, 1);     // CNOT: control=0, target=1
qc.Z(1);           // Z gate on qubit 1
```

All gate methods call the private `Add` helper:

```csharp
private void Add(string label, GateMatrix matrix, params int[] qubits)
{
    _ops.Add(new GateOp(label, matrix, qubits));
    _history.Add($"{label}({string.Join(",", qubits)})");
}
```

### Execution

```csharp
QuantumState state = qc.Run();
```

`Run()` creates a fresh `QuantumState` in `|0…0⟩` and applies each recorded gate in order:

```csharp
foreach (var op in _ops)
{
    switch (op.Qubits.Length)
    {
        case 1: Gates.ApplySingle(state, op.Matrix, op.Qubits[0]); break;
        case 2: Gates.ApplyTwo   (state, op.Matrix, op.Qubits[0], op.Qubits[1]); break;
        case 3: Gates.ApplyThree (state, op.Matrix, op.Qubits[0], op.Qubits[1], op.Qubits[2]); break;
    }
}
```

Because `Run()` always starts from a fresh state, the same circuit object can be executed multiple times with no side effects:

```csharp
// Run the same Bell-state circuit with different starting conditions
var state0 = qc.Run(initialState: 0);   // |00⟩ input
var state3 = qc.Run(initialState: 3);   // |11⟩ input
```

---

## 6. Measurement

**Methods:** `QuantumState.Measure`, `QuantumState.MeasureQubit`

### Full measurement — `Measure(shots)`

Measurement is the **irreversible** process by which a quantum state collapses into one of the classical basis states.

**Before:** `|ψ⟩ = Σ αᵢ|i⟩` — all basis states coexist.  
**After:**  system found in state `|k⟩` with probability `P(k) = |αₖ|²`.

The implementation uses the **inverse-CDF method** (also called the roulette-wheel or stochastic acceptance method):

```csharp
double r = Rng.NextDouble();       // uniform random number in [0, 1)
double cumulative = 0.0;
int outcome = NStates - 1;

for (int i = 0; i < NStates; i++)
{
    cumulative += probs[i];
    if (r < cumulative)
    {
        outcome = i;
        break;
    }
}
```

Walk along the cumulative distribution until the running sum exceeds `r`. The first index where this happens is the outcome. This is mathematically identical to the Born rule: basis state `i` is selected with probability `probs[i]`.

After sampling, the state collapses:
```csharp
Array.Clear(Amplitudes, 0, NStates);
Amplitudes[outcome] = Complex.One;
```

The random seed is fixed at 42 (`new Random(42)`) for reproducibility across runs.

### Partial measurement — `MeasureQubit(qubit)`

Measures a **single qubit** and leaves the rest of the system in a renormalised conditional state.

```csharp
// 1. Sum |αᵢ|² for all basis states where this qubit = 1
double prob1 = 0;
for (int i = 0; i < NStates; i++)
    if (((i >> bitPos) & 1) == 1)
        prob1 += Amplitudes[i].MagnitudeSquared();

// 2. Sample outcome
int outcome = Rng.NextDouble() < (1 - prob1) ? 0 : 1;

// 3. Zero out inconsistent amplitudes, then renormalise
```

This is used in the entanglement correlation demo (Demo 2) and the teleportation demo (Demo 7) to show that measuring one qubit of an entangled pair instantly collapses the state of the other.

---

## 7. Visualiser.cs

**Static class:** `Visualiser`

Renders the probability distribution as an ASCII bar chart. No external library needed.

```csharp
Visualiser.Print(state, "Bell state |Φ+⟩");
```

Example output:
```
════════════════════════════════════════════════════════════════
  Bell state |Φ+⟩
════════════════════════════════════════════════════════════════
  |00⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
  |01⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |10⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
  |11⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
```

The bar width is 40 characters. Each character represents 2.5% probability. Filled blocks (`█`) show the probability; empty blocks (`░`) fill the remainder to 100%.

`AsciiBar()` returns the chart as a `string`; `Print()` writes it to `Console`.

---

## 8. Demo 1 — Single Qubit

**Function:** `Program.cs` — top-level statements, first section

### Initial state `|0⟩`
```csharp
var q = new QuantumState(1);
// Amplitudes = [1+0i, 0+0i]
// 100% probability of measuring 0
```

### After `X` gate → `|1⟩`
```csharp
Gates.ApplySingle(q, Gates.X(), 0);
// Amplitudes = [0+0i, 1+0i]
// The X matrix swaps the two amplitudes
```

### After `H` gate → `|+⟩ = (|0⟩+|1⟩)/√2`
```csharp
Gates.ApplySingle(q, Gates.H(), 0);
// Amplitudes = [0.707+0i, 0.707+0i]
// Measuring gives 0 or 1 with equal 50% probability
```

### `H` twice = Identity
Applying `H` twice cancels out — the second H causes the two equal amplitudes to **interfere**, reconstructing the original state. This interference is the foundation of quantum computing.

### `Z` gate on `|+⟩` → `|−⟩`
```csharp
// After Z: Amplitudes = [0.707+0i, -0.707+0i]
// Probabilities unchanged — but phase of |1⟩ has flipped
```
The probabilities are identical to `|+⟩`. The phase flip is invisible to direct measurement but becomes observable when further gates are applied.

### Measurement — 20 independent shots
Each shot creates a fresh `|+⟩` state and measures it. Outputs approximately 10 zeros and 10 ones, demonstrating the Born rule stochastically.

---

## 9. Demo 2 — Two Qubits and Entanglement

**Second section of Program.cs**

### Product state — `H` on qubit 0 only
```
|ψ⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
```
Qubit 1 is definitely `|0⟩`. Measuring qubit 1 always gives 0, regardless of qubit 0. This is a **product state** — the qubits are independent.

### Bell state `|Φ+⟩` — circuit: `H(0)` then `CNOT(0, 1)`

Step by step:
1. Start: `|00⟩`
2. `H(0)`: `(|00⟩ + |10⟩)/√2`
3. `CNOT(0,1)`: `(|00⟩ + |11⟩)/√2`

```
  |00⟩  ████████████████████  50.00%
  |01⟩                         0.00%
  |10⟩                         0.00%
  |11⟩  ████████████████████  50.00%
```

States `|01⟩` and `|10⟩` have **zero probability**. This state cannot be written as a product — the qubits are **entangled**.

### Entanglement correlation demo

```csharp
for (int trial = 1; trial <= 10; trial++)
{
    var s = QuantumAlgorithms.Bell(QuantumAlgorithms.BellState.PhiPlus).State;
    int r0 = s.MeasureQubit(0);
    int r1 = s.MeasureQubit(1);
    // r0 == r1 every single time
}
```

`MeasureQubit(0)` collapses the joint state. The second call `MeasureQubit(1)` then observes the already-collapsed state — always giving the same result as qubit 0.

### All four Bell states

```csharp
QuantumAlgorithms.Bell(BellState.PhiPlus)   // (|00⟩ + |11⟩)/√2
QuantumAlgorithms.Bell(BellState.PhiMinus)  // (|00⟩ - |11⟩)/√2
QuantumAlgorithms.Bell(BellState.PsiPlus)   // (|01⟩ + |10⟩)/√2
QuantumAlgorithms.Bell(BellState.PsiMinus)  // (|01⟩ - |10⟩)/√2
```

Each uses a combination of `X`, `H`, `Z`, and `CNOT` gates. The `BellState` enum makes call sites self-documenting.

---

## 10. Demo 3 — Three Qubits and the Toffoli Gate

**Third section of Program.cs**

### GHZ state

```csharp
var ghz = QuantumAlgorithms.GHZ(3);
// State: (|000⟩ + |111⟩)/√2
// Only two non-zero amplitudes out of 8
```

Circuit: `H(0)`, `CNOT(0,1)`, `CNOT(1,2)`.

All three qubits are entangled — measuring any one determines all three.

### Toffoli gate — truth table demo

```csharp
var qs = new QuantumState(3, 0b110);    // |110⟩
Gates.ApplyThree(qs, Gates.Toffoli(), 0, 1, 2);
// Result: |111⟩  — both controls were 1, target flipped
```

```
Toffoli |110⟩ → |111⟩   (c0=1, c1=1 → target 0→1)
Toffoli |111⟩ → |110⟩   (c0=1, c1=1 → target 1→0)
Toffoli |010⟩ → |010⟩   (c0=0       → target unchanged)
```

---

## 11. Demo 4 — Four Qubits

**Fourth section of Program.cs**

### Uniform superposition

```csharp
var qc = new QuantumCircuit(4);
for (int i = 0; i < 4; i++) qc.H(i);
var state = qc.Run();
// All 16 amplitudes = 0.25 + 0i
// All 16 probabilities = 6.25%
```

### 4-qubit GHZ

```csharp
var ghz4 = QuantumAlgorithms.GHZ(4);
// State: (|0000⟩ + |1111⟩)/√2
// 2 non-zero states out of 16
```

---

## 12. Demo 5 — Deutsch-Jozsa Algorithm

**`QuantumAlgorithms.DeutschJozsa(nInput, oracle)`**

### The problem

Given a black-box function `f: {0,1}ⁿ → {0,1}` that is guaranteed to be either **constant** (same output for all inputs) or **balanced** (0 for exactly half the inputs), determine which.

- Classical worst case: up to `2^(n-1)+1` evaluations
- Quantum: **exactly 1 evaluation** — exponential speedup

### The oracle enum

```csharp
QuantumAlgorithms.DeutschJozsa(nInput: 2, oracle: DJOracle.Constant0)
QuantumAlgorithms.DeutschJozsa(nInput: 2, oracle: DJOracle.Constant1)
QuantumAlgorithms.DeutschJozsa(nInput: 2, oracle: DJOracle.Balanced)
```

### Circuit walkthrough (n=2, Balanced oracle)

```csharp
qc.X(ancilla);          // put ancilla in |1⟩

qc.H(0); qc.H(1); qc.H(2);   // H on all: input in superposition, ancilla in |−⟩

// Balanced oracle: CNOT from each input qubit to ancilla
// Phase kick-back: (-1)^f(x)|x⟩|−⟩ encodes f as a ±1 phase
qc.CNOT(0, ancilla);
qc.CNOT(1, ancilla);

qc.H(0); qc.H(1);      // H on inputs → interference
// Measure: any non-zero result → BALANCED
```

### What the simulator shows

| Oracle | Expected | P(input register = |00⟩) |
|--------|----------|--------------------------|
| `Constant0` | CONSTANT | 100% |
| `Constant1` | CONSTANT | 100% |
| `Balanced`  | BALANCED | 0%   |

---

## 13. Demo 6 — Grover's Search Algorithm

**`QuantumAlgorithms.Grover(nQubits, target)`**

### The problem

Find a marked item in an unstructured list of `N = 2ⁿ` items.

- Classical: O(N) queries
- Quantum: O(√N) queries — **quadratic speedup**

### Running it

```csharp
var result = QuantumAlgorithms.Grover(nQubits: 2, target: 2);
// Searches 4 items, marks |10⟩, runs 1 iteration
// P(|10⟩) ≈ 1.0 after algorithm

var result4 = QuantumAlgorithms.Grover(nQubits: 4, target: 9);
// Searches 16 items, marks |1001⟩, runs 3 iterations
// P(|1001⟩) ≈ 0.96 after algorithm
```

### Iterations computed automatically

```csharp
int nIterations = Math.Max(1, (int)Math.Round(Math.PI / 4.0 * Math.Sqrt(N)));
```

This is the optimal number of iterations for the given `N`.

### One iteration structure

```
1. Oracle (phase flip of |target⟩):
   - X on qubits where target bit = 0   (map target to |1…1⟩)
   - Multi-controlled Z                  (flip phase of |1…1⟩)
   - X on same qubits                   (undo the mapping)

2. Diffusion (inversion about the mean):
   - H on all qubits
   - X on all qubits
   - Multi-controlled Z
   - X on all qubits
   - H on all qubits
```

### Multi-controlled Z implementation

The private `ApplyMultiControlledZ` helper handles 1–4 qubits:

```csharp
case 2: qc.H(1); qc.CNOT(0, 1); qc.H(1);            // H·CNOT·H = CZ
case 3: qc.H(2); qc.Toffoli(0, 1, 2); qc.H(2);       // H·CCX·H = CCZ
case 4: qc.H(3); qc.Toffoli(0,1,2); qc.CNOT(2,3);    // decomposed
        qc.Toffoli(0,1,2); qc.H(3);
```

### Demo results

| n | N | Target | Iterations | P(target) |
|---|---|--------|-----------|-----------|
| 2 | 4 | `\|10⟩`   | 1 | ~100% |
| 3 | 8 | `\|101⟩`  | 2 | ~97%  |
| 4 | 16| `\|1001⟩` | 3 | ~96%  |

The probability is not exactly 100% because the algorithm slightly overshoots after the optimal number of iterations.

---

## 14. Demo 7 — Quantum Teleportation

**`QuantumAlgorithms.Teleportation()`**

### The protocol (3 qubits)

- **q0** — message qubit (Alice): prepared in `Ry(π/3)|0⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩`
- **q1** — Alice's half of the Bell pair
- **q2** — Bob's half of the Bell pair

```csharp
qc.Ry(0, Math.PI / 3.0);   // prepare unknown message state

qc.H(1);                    // create Bell pair between q1 and q2
qc.CNOT(1, 2);

qc.CNOT(0, 1);              // Alice's Bell-basis rotation
qc.H(0);
```

The demo shows the joint 3-qubit state after step 4 (before Alice measures). In a real implementation:
1. Alice measures q0 and q1, obtaining 2 classical bits `b₀, b₁`
2. Bob applies `X` to q2 if `b₁ = 1`, then `Z` to q2 if `b₀ = 1`
3. q2 is now exactly in state `|ψ⟩`

Key point: the 2 classical bits must be sent conventionally — teleportation does **not** allow faster-than-light communication.

---

## 15. Demo 8 — Phase Gates and Interference

**Final section of Program.cs**

Circuit for each phase gate: `|0⟩ → H → [gate] → H → measure`

```csharp
foreach (var (gateFn, name, phaseLabel) in new (Func<GateMatrix>, string, string)[]
{
    (() => Gates.Z(), "Z", "+π"),
    (() => Gates.S(), "S", "+π/2"),
    (() => Gates.T(), "T", "+π/4"),
})
{
    var q = new QuantumState(1);
    Gates.ApplySingle(q, Gates.H(), 0);
    Gates.ApplySingle(q, gateFn(), 0);
    Gates.ApplySingle(q, Gates.H(), 0);
    Console.WriteLine(q);
}
```

| Gate | Phase added to `\|1⟩` | P(`\|0⟩`) after H·gate·H | P(`\|1⟩`) after H·gate·H |
|------|----------------------|--------------------------|--------------------------|
| Z    | +π (180°)            | 0%                       | 100%                     |
| S    | +π/2 (90°)           | 50%                      | 50%                      |
| T    | +π/4 (45°)           | 85.4%                    | 14.6%                    |

The same input `|+⟩` produces completely different measurement outcomes depending on the phase applied. This demonstrates how quantum algorithms encode information in phases and extract it through controlled interference.

---

## 16. Key C# vs Python Differences

### No external math library

Python uses NumPy for complex arithmetic, array reshaping, and `tensordot`. C# uses only `System.Numerics.Complex` from the BCL. The applicators implement the tensor contraction manually via explicit pair-iteration loops rather than matrix-vector products.

| Concern | Python (NumPy) | C# (.NET 8) |
|---------|---------------|-------------|
| Complex type | `np.complex128` | `System.Numerics.Complex` |
| Amplitude array | `np.ndarray` | `Complex[]` |
| Gate matrix | `np.ndarray` (2D) | `GateMatrix` struct (flat row-major `Complex[]`) |
| Gate application | `np.tensordot` + `reshape` | Explicit pair-iteration loops |
| Random sampling | `np.random.choice(p=probs)` | Inverse-CDF walk |
| Squared magnitude | `np.abs(a)**2` | `a.Real*a.Real + a.Imaginary*a.Imaginary` |

### GateMatrix struct vs ndarray

In Python, a gate is just a 2D NumPy array. In C# it is a `GateMatrix` readonly struct that carries its `Dim` and `Arity` alongside the data, providing a `this[row, col]` indexer. This makes the intent explicit and catches arity mismatches at construction time.

### Measurement (inverse-CDF vs `np.random.choice`)

NumPy's `random.choice(p=probs)` handles the weighted sampling internally. In C#, the equivalent is an explicit cumulative-sum walk — mathematically identical but written out longhand.

### Enum parameters vs strings

Where Python uses string arguments (`oracle_type="balanced"`, `which="phi+"`), C# uses enums (`DJOracle.Balanced`, `BellState.PhiPlus`). This eliminates typo bugs and enables IDE autocompletion.

### `file` modifier on internal types

The `GateOp` record in `QuantumCircuit.cs` uses the C# 11 `file` access modifier, making it invisible outside that file without polluting the namespace with a `private` nested type.

### `stackalloc` in ApplyThree

The eight-index buffer in `ApplyThree` uses `Span<int> idx = stackalloc int[8]` to allocate on the stack rather than the heap, avoiding per-group GC pressure in the inner loop.

---

## 17. Mathematical Reference

### Dirac (Ket) Notation

| Symbol | Meaning |
|--------|---------|
| `\|ψ⟩` | A quantum state vector |
| `⟨ψ\|` | The conjugate transpose (bra) |
| `⟨φ\|ψ⟩` | Inner product between two states |
| `\|ψ⟩⊗\|φ⟩` | Tensor product (composite system) |
| `\|0⟩, \|1⟩` | Computational basis states |

### Computational Basis Index Mapping

For n qubits, index `i` (0 to `2ⁿ-1`) maps to the basis state via the binary representation of `i`, with qubit 0 as the most significant bit.

In C#:
```csharp
string label = $"|{Convert.ToString(i, 2).PadLeft(n, '0')}⟩";
```

### Born Rule

```
P(i) = |αᵢ|²  =  αᵢ.Real² + αᵢ.Imaginary²
```

All probabilities sum to 1: `Σ P(i) = 1`

In C#:
```csharp
double prob = amp.Real * amp.Real + amp.Imaginary * amp.Imaginary;
```

### Unitarity

Every quantum gate `U` satisfies `U†U = I`.  
In C#, this is enforced implicitly — all gate matrices are hand-verified to be unitary, and the `ApplySingle`/`ApplyTwo`/`ApplyThree` methods preserve norm exactly.

### Tensor Product

For two state vectors `|ψ⟩ ∈ ℂᵐ` and `|φ⟩ ∈ ℂⁿ`:

```
(|ψ⟩ ⊗ |φ⟩)[i·n + j] = ψ[i] × φ[j]
```

The flat index `i·n + j` is why a 2-qubit system has indices 0–3 (`2×2`), a 3-qubit system 0–7 (`2×2×2`), and so on.

---

## Running the Full Demo Suite

```bash
cd /path/to/QuantumSimulation
dotnet run --project csharp/QuantumSimulator
```

The program runs all 8 demos in sequence and prints the full output to stdout. Pipe to a file to capture it:

```bash
dotnet run --project csharp/QuantumSimulator > output.txt
```

Expected runtime: under 1 second for all demos on any modern machine.
