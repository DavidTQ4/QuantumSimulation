// =============================================================================
//  Program.cs — Demo runner
//  Entry point for the Four-Qubit Quantum Computing Simulator (C# edition).
// =============================================================================

using QuantumSimulator;

// ── Header ───────────────────────────────────────────────────────────────────

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          FOUR-QUBIT QUANTUM COMPUTING SIMULATOR  (C# / .NET 8)             ║
║          A fully annotated educational implementation                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  This simulation is EXACT: it uses the full 2ⁿ-dimensional state vector
  and represents every gate as a unitary matrix.  No approximations are made.
  Measurement is simulated stochastically using the Born rule.
""");

// ── Demo 1 — Single Qubit ────────────────────────────────────────────────────
Section("DEMO 1 — Single Qubit: Basis, Superposition, Phase, Measurement");

Console.WriteLine("""
  A single qubit is described by a 2-element complex vector:
      |ψ⟩ = α|0⟩ + β|1⟩,   where |α|² + |β|² = 1

  |0⟩ and |1⟩ are the computational basis states (like classical 0 and 1).
  α and β are complex AMPLITUDES; their squared magnitudes give probabilities.
""");

// Initial state |0⟩
Console.WriteLine("  ┌─ Initial state |0⟩ ───────────────────────────────────────────┐");
var q = new QuantumState(1);
Console.WriteLine(q);
Visualiser.Print(q, "|0⟩");

// X gate (NOT)
Console.WriteLine("  ┌─ After X gate (NOT): |0⟩ → |1⟩ ──────────────────────────────┐");
q = new QuantumState(1);
Gates.ApplySingle(q, Gates.X(), 0);
Console.WriteLine(q);
Visualiser.Print(q, "X|0⟩ = |1⟩");

// H gate — equal superposition
Console.WriteLine("  ┌─ After H gate: |0⟩ → (|0⟩+|1⟩)/√2  (equal superposition) ───┐");
q = new QuantumState(1);
Gates.ApplySingle(q, Gates.H(), 0);
Console.WriteLine(q);
Visualiser.Print(q, "H|0⟩ = |+⟩");

// H twice = Identity
Console.WriteLine("  H applied twice = Identity (H is its own inverse):");
Gates.ApplySingle(q, Gates.H(), 0);
Console.WriteLine(q);

// Z gate — phase flip
Console.WriteLine("  ┌─ Z gate on |+⟩: creates |−⟩ = (|0⟩−|1⟩)/√2 ─────────────────┐");
q = new QuantumState(1);
Gates.ApplySingle(q, Gates.H(), 0);
Gates.ApplySingle(q, Gates.Z(), 0);
Console.WriteLine(q);
Visualiser.Print(q, "ZH|0⟩ = |−⟩");

// S and T gates
Console.WriteLine("  ┌─ S gate (π/2 phase) and T gate (π/4 phase) on |+⟩ ───────────┐");
foreach (var (gateFn, name) in new (Func<GateMatrix> GateFn, string Name)[]
    { (() => Gates.S(), "S"), (() => Gates.T(), "T") })
{
    q = new QuantumState(1);
    Gates.ApplySingle(q, Gates.H(), 0);
    Gates.ApplySingle(q, gateFn(), 0);
    Console.WriteLine($"  {name} gate on |+⟩:");
    Console.WriteLine(q);
}

// Measurement demo — 20 shots of |+⟩
Console.WriteLine("  ┌─ Measurement of superposition: 20 shots of H|0⟩ ──────────────┐");
Console.WriteLine("""
  We prepare |+⟩ = (|0⟩+|1⟩)/√2 and measure 20 times independently.
  Expected: ~10 zeros, ~10 ones.
""");
var results = new List<int>();
for (int i = 0; i < 20; i++)
{
    q = new QuantumState(1);
    Gates.ApplySingle(q, Gates.H(), 0);
    results.Add(q.Measure()[0]);
}
Console.WriteLine($"  20 measurements: [{string.Join(", ", results)}]");
Console.WriteLine($"  Count(0) = {results.Count(r => r == 0)},  Count(1) = {results.Count(r => r == 1)}");

// ── Demo 2 — Two Qubits ──────────────────────────────────────────────────────
Section("DEMO 2 — Two Qubits: Product States, CNOT, Entanglement");

Console.WriteLine("""
  A 2-qubit system has 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩.
  The state vector has 4 complex amplitudes.

  A PRODUCT STATE can be written as (state of q0) ⊗ (state of q1).
  An ENTANGLED STATE cannot — measuring one qubit determines the other.
""");

// Product state: H on q0 only
Console.WriteLine("  ┌─ Product state: H on q0, nothing on q1 ───────────────────────┐");
var q2 = new QuantumState(2);
Gates.ApplySingle(q2, Gates.H(), 0);
Console.WriteLine(q2);
Visualiser.Print(q2, "(|0⟩+|1⟩)/√2 ⊗ |0⟩");

// Bell state |Φ+⟩
Console.WriteLine("  ┌─ Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 ──────────────────────────┐");
var bell = QuantumAlgorithms.Bell(QuantumAlgorithms.BellState.PhiPlus);
Console.WriteLine($"  {bell.Description}");
Console.WriteLine(bell.State);
Visualiser.Print(bell.State, "Bell state |Φ+⟩");

// Entanglement correlation: 10 trials
Console.WriteLine("  ┌─ Entanglement correlation: measure q0, observe q1 ────────────┐");
Console.WriteLine("""
  Prepare |Φ+⟩ 10 times.  Measure q0 then q1.
  Perfect entanglement: outcomes always agree (both 0 or both 1).
""");
for (int trial = 1; trial <= 10; trial++)
{
    var s = QuantumAlgorithms.Bell(QuantumAlgorithms.BellState.PhiPlus).State;
    int r0 = s.MeasureQubit(0);
    int r1 = s.MeasureQubit(1);
    string match = r0 == r1 ? "✓" : "✗";
    Console.WriteLine($"    Trial {trial,2}: q0={r0},  q1={r1}  {match}");
}

// All four Bell states
Console.WriteLine("\n  ┌─ All four Bell states ──────────────────────────────────────────┐");
foreach (var bs in Enum.GetValues<QuantumAlgorithms.BellState>())
{
    var r = QuantumAlgorithms.Bell(bs);
    Visualiser.Print(r.State, bs.ToString());
}

// ── Demo 3 — Three Qubits ────────────────────────────────────────────────────
Section("DEMO 3 — Three Qubits: GHZ State and Toffoli Gate");

Console.WriteLine("""
  3 qubits → 8 basis states: |000⟩ through |111⟩.
  State vector has 8 complex amplitudes.
""");

// GHZ state
var ghz3 = QuantumAlgorithms.GHZ(3);
Console.WriteLine(ghz3.Description);
Console.WriteLine(ghz3.State);
Visualiser.Print(ghz3.State, "3-qubit GHZ = (|000⟩ + |111⟩)/√2");

// Toffoli truth table
Console.WriteLine("  ┌─ Toffoli (CCNOT): flips target iff both controls = 1 ──────────┐");
Console.WriteLine("""
  |control0, control1, target⟩ → target flipped iff c0 AND c1 = 1

    |110⟩ → |111⟩   (both controls 1, target 0→1)
    |111⟩ → |110⟩   (both controls 1, target 1→0)
    |010⟩ → |010⟩   (only one control 1, unchanged)
""");
foreach (int init in new[] { 0b110, 0b111, 0b010 })
{
    var qs = new QuantumState(3, init);
    Gates.ApplyThree(qs, Gates.Toffoli(), 0, 1, 2);
    double[] p = qs.Probabilities();
    int result = 0;
    for (int i = 1; i < 8; i++)
        if (p[i] > p[result]) result = i;
    Console.WriteLine($"  Toffoli |{Convert.ToString(init, 2).PadLeft(3, '0')}⟩ → |{Convert.ToString(result, 2).PadLeft(3, '0')}⟩");
}

// ── Demo 4 — Four Qubits ─────────────────────────────────────────────────────
Section("DEMO 4 — Four Qubits: Full 16-State Hilbert Space");

Console.WriteLine("""
  4 qubits → 16 basis states: |0000⟩ through |1111⟩.
  State vector has 16 complex amplitudes.

  H⊗H⊗H⊗H |0000⟩ creates a UNIFORM superposition over all 16 states,
  each with amplitude 1/4 and probability 1/16 = 6.25%.
""");

var qc4 = new QuantumCircuit(4);
for (int i = 0; i < 4; i++) qc4.H(i);
var state4 = qc4.Run();
Console.WriteLine(state4);
Visualiser.Print(state4, "H⊗H⊗H⊗H |0000⟩ — uniform 4-qubit superposition");

Console.WriteLine("  ┌─ 4-qubit GHZ state ────────────────────────────────────────────┐");
var ghz4 = QuantumAlgorithms.GHZ(4);
Console.WriteLine(ghz4.Description);
Console.WriteLine(ghz4.State);
Visualiser.Print(ghz4.State, "4-qubit GHZ = (|0000⟩ + |1111⟩)/√2");

// ── Demo 5 — Deutsch-Jozsa ───────────────────────────────────────────────────
Section("DEMO 5 — Deutsch-Jozsa Algorithm");

Console.WriteLine("""
  Problem: is f constant (same output for all inputs)
           or balanced (0 for half the inputs, 1 for the other half)?

  Classical worst case: 2^(n-1)+1 evaluations.
  Quantum: exactly 1 evaluation (exponential speedup).

  Measuring all-zero input register → CONSTANT.
  Any non-zero result               → BALANCED.
""");

foreach (var oracle in Enum.GetValues<QuantumAlgorithms.DJOracle>())
{
    var dj = QuantumAlgorithms.DeutschJozsa(nInput: 2, oracle: oracle);
    Console.WriteLine($"\n  Oracle: {oracle}");
    Console.WriteLine($"  {dj.Description}");
    Visualiser.Print(dj.State, $"Deutsch-Jozsa ({oracle})");
}

// ── Demo 6 — Grover's Search ─────────────────────────────────────────────────
Section("DEMO 6 — Grover's Quantum Search Algorithm");

Console.WriteLine("""
  Problem: find a marked item in an unstructured list of N items.
  Classical: O(N) queries.  Quantum: O(√N) queries — quadratic speedup.

  The algorithm amplifies the amplitude of the target state so that
  measuring gives the target with high probability.
""");

var groverCases = new (int N, int Target, string Label)[]
{
    (2, 2,  "4 items,  target |10⟩"),
    (2, 3,  "4 items,  target |11⟩"),
    (3, 5,  "8 items,  target |101⟩"),
    (4, 9,  "16 items, target |1001⟩"),
    (4, 14, "16 items, target |1110⟩"),
};

foreach (var (n, t, label) in groverCases)
{
    var g = QuantumAlgorithms.Grover(nQubits: n, target: t);
    Console.WriteLine($"\n  {label}");
    Console.WriteLine($"  {g.Description}");
    Visualiser.Print(g.State, $"Grover n={n} target={t}");
}

// ── Demo 7 — Teleportation ───────────────────────────────────────────────────
Section("DEMO 7 — Quantum Teleportation");

Console.WriteLine("""
  Teleportation transfers an unknown quantum state |ψ⟩ from Alice to Bob
  using one pre-shared Bell pair + two classical bits.

  It does NOT allow faster-than-light communication: the 2 classical bits
  must still travel at light speed.  But the quantum information (the exact
  α, β of |ψ⟩) is transferred perfectly, without ever being directly measured.
""");

var tel = QuantumAlgorithms.Teleportation();
Console.WriteLine(tel.Description);
Console.WriteLine(tel.State);
Visualiser.Print(tel.State, "Teleportation — joint state before Alice measures");

// ── Demo 8 — Phase Gates ─────────────────────────────────────────────────────
Section("DEMO 8 — Phase Gates: Z, S, T Acting on Superposition");

Console.WriteLine("""
  Phase gates leave |0⟩ unchanged but rotate the phase of |1⟩:
      Z: adds +π  (180°) — multiplies |1⟩ amplitude by -1
      S: adds +π/2 (90°) — multiplies |1⟩ amplitude by i
      T: adds +π/4 (45°) — multiplies |1⟩ amplitude by e^(iπ/4)

  A phase change is invisible to direct measurement (|αe^(iφ)|² = |α|²)
  but creates INTERFERENCE when combined with other gates.
  Circuit: |0⟩ → H → [phase gate] → H → measure
""");

foreach (var (gateFn, name, phaseLabel) in new (Func<GateMatrix> GateFn, string Name, string Phase)[]
{
    (() => Gates.Z(), "Z", "+π"),
    (() => Gates.S(), "S", "+π/2"),
    (() => Gates.T(), "T", "+π/4"),
})
{
    q = new QuantumState(1);
    Gates.ApplySingle(q, Gates.H(), 0);     // create |+⟩
    Gates.ApplySingle(q, gateFn(), 0);      // apply phase gate
    Gates.ApplySingle(q, Gates.H(), 0);     // interfere back
    Console.WriteLine($"  H → {name}({phaseLabel}) → H  on |0⟩:");
    Console.WriteLine(q);
    Visualiser.Print(q, $"H {name} H |0⟩");
}

// ── Summary ──────────────────────────────────────────────────────────────────
Section("SUMMARY — WHAT WE COVERED");

Console.WriteLine("""
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

  All results are EXACT quantum mechanical predictions.
""");

// ── Helpers ───────────────────────────────────────────────────────────────────

static void Section(string title)
{
    Console.WriteLine();
    Console.WriteLine(new string('═', 70));
    Console.WriteLine($"  {title}");
    Console.WriteLine(new string('═', 70));
    Console.WriteLine();
}
