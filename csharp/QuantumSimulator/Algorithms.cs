// =============================================================================
//  Algorithms.cs — landmark quantum algorithms
// =============================================================================
//
//  Each method builds and runs a QuantumCircuit, then returns a result record
//  containing the circuit, the final state, and a human-readable description.
// =============================================================================

namespace QuantumSimulator;

// ---------------------------------------------------------------------------
//  Result container returned by every algorithm.
// ---------------------------------------------------------------------------

public sealed record AlgorithmResult(
    QuantumCircuit Circuit,
    QuantumState   State,
    string         Description
);

// ---------------------------------------------------------------------------
//  Algorithms
// ---------------------------------------------------------------------------

public static class QuantumAlgorithms
{
    // =========================================================================
    //  BELL STATES
    //
    //  The four Bell (EPR) states are the maximally entangled 2-qubit states.
    //  They form an orthonormal basis for the 4-dimensional 2-qubit Hilbert
    //  space — the "Bell basis".
    //
    //  |Φ+⟩ = (|00⟩ + |11⟩)/√2    BellState.PhiPlus   (default)
    //  |Φ-⟩ = (|00⟩ - |11⟩)/√2    BellState.PhiMinus
    //  |Ψ+⟩ = (|01⟩ + |10⟩)/√2    BellState.PsiPlus
    //  |Ψ-⟩ = (|01⟩ - |10⟩)/√2    BellState.PsiMinus
    //
    //  CIRCUIT FOR |Φ+⟩
    //  ─────────────────
    //    q0: ─ H ─●─
    //              │
    //    q1: ─────X─
    //
    //  Step 1: H on q0 → (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩+|10⟩)/√2
    //  Step 2: CNOT     → (|00⟩+|11⟩)/√2   ← entangled!
    //
    //  ENTANGLEMENT TEST
    //  -----------------
    //  After |Φ+⟩: measure q0.
    //    • q0 = 0 → q1 collapses to |0⟩ — always 0.
    //    • q0 = 1 → q1 collapses to |1⟩ — always 1.
    //  Perfect correlation regardless of physical separation.
    // =========================================================================

    public enum BellState { PhiPlus, PhiMinus, PsiPlus, PsiMinus }

    /// <summary>Prepare one of the four Bell states on a 2-qubit circuit.</summary>
    public static AlgorithmResult Bell(BellState which = BellState.PhiPlus)
    {
        var qc = new QuantumCircuit(2);

        // Ψ states need q1 flipped to |1⟩ first.
        if (which is BellState.PsiPlus or BellState.PsiMinus)
            qc.X(1);

        // H on q0 creates the superposition.
        qc.H(0);

        // Φ- and Ψ- need a Z to flip the sign of the |1⟩ component.
        if (which is BellState.PhiMinus or BellState.PsiMinus)
            qc.Z(0);

        // CNOT entangles the two qubits.
        qc.CNOT(0, 1);

        var state = qc.Run();

        string name = which switch
        {
            BellState.PhiPlus  => "|Φ+⟩ = (|00⟩ + |11⟩)/√2",
            BellState.PhiMinus => "|Φ-⟩ = (|00⟩ - |11⟩)/√2",
            BellState.PsiPlus  => "|Ψ+⟩ = (|01⟩ + |10⟩)/√2",
            BellState.PsiMinus => "|Ψ-⟩ = (|01⟩ - |10⟩)/√2",
            _                  => ""
        };

        string desc =
            $"Bell state {name}\n" +
            $"  {qc.Diagram()}\n" +
            $"  Maximally entangled: measuring either qubit collapses both.";

        return new AlgorithmResult(qc, state, desc);
    }

    // =========================================================================
    //  GHZ STATE
    //
    //  The n-qubit Greenberger–Horne–Zeilinger state:
    //
    //      |GHZ_n⟩ = (|00…0⟩ + |11…1⟩)/√2
    //
    //  All n qubits are perfectly correlated: measuring any one determines all.
    //
    //  CIRCUIT
    //  -------
    //    q0: ─ H ─●────────
    //              │
    //    q1: ─────X─●──────
    //                │
    //    q2: ─────────X────
    //    (and so on)
    // =========================================================================

    /// <summary>Prepare the n-qubit GHZ state (n ≥ 2).</summary>
    public static AlgorithmResult GHZ(int nQubits = 3)
    {
        if (nQubits < 2)
            throw new ArgumentOutOfRangeException(nameof(nQubits), "Must be ≥ 2.");

        var qc = new QuantumCircuit(nQubits);
        qc.H(0);
        for (int i = 0; i < nQubits - 1; i++)
            qc.CNOT(i, i + 1);

        var state = qc.Run();
        string zeros = new('0', nQubits);
        string ones  = new('1', nQubits);

        string desc =
            $"{nQubits}-qubit GHZ state: (|{zeros}⟩ + |{ones}⟩)/√2\n" +
            $"  All {nQubits} qubits maximally entangled.";

        return new AlgorithmResult(qc, state, desc);
    }

    // =========================================================================
    //  DEUTSCH-JOZSA ALGORITHM
    //
    //  PROBLEM
    //  -------
    //  Given f: {0,1}ⁿ → {0,1} that is guaranteed to be either:
    //    • CONSTANT  — same output for every input, OR
    //    • BALANCED  — 0 for half the inputs, 1 for the other half.
    //  Determine which.
    //
    //  Classical worst case: up to 2^(n-1)+1 evaluations.
    //  Quantum: exactly ONE evaluation.  Exponential speedup.
    //
    //  CIRCUIT  (n=2 input qubits + 1 ancilla)
    //  ─────────────────────────────────────────
    //    q0: ─ H ────[Oracle]── H ──M
    //    q1: ─ H ────[Oracle]── H ──M
    //    q2: ─ X ─ H ─[Oracle]──────
    //
    //  PHASE KICK-BACK TRICK
    //  ---------------------
    //  With the ancilla in |−⟩ = (|0⟩-|1⟩)/√2:
    //      Uf|x⟩|−⟩ = (-1)^f(x) |x⟩|−⟩
    //  The oracle writes f(x) as a ±1 PHASE on the input, not into a qubit.
    //  After final H gates: phases interfere.
    //    Constant → |00…0⟩ with certainty.
    //    Balanced → anything except |00…0⟩.
    // =========================================================================

    public enum DJOracle { Constant0, Constant1, Balanced }

    /// <summary>
    /// Run the Deutsch-Jozsa algorithm.
    /// </summary>
    /// <param name="nInput">Number of input qubits (total = nInput + 1 ancilla).</param>
    /// <param name="oracle">Which oracle to embed in the circuit.</param>
    public static AlgorithmResult DeutschJozsa(int nInput = 2, DJOracle oracle = DJOracle.Balanced)
    {
        int nTotal  = nInput + 1;
        int ancilla = nInput;     // ancilla is the last qubit

        var qc = new QuantumCircuit(nTotal);

        // Step 1: put ancilla in |1⟩
        qc.X(ancilla);

        // Step 2: H on all qubits (creates superposition + ancilla in |−⟩)
        for (int i = 0; i < nTotal; i++)
            qc.H(i);

        // Step 3: Oracle
        switch (oracle)
        {
            case DJOracle.Constant0:
                // f(x) = 0 for all x → identity oracle (no gates needed)
                break;

            case DJOracle.Constant1:
                // f(x) = 1 for all x → X on ancilla for every input.
                // This is equivalent to a global -1 phase: undetectable.
                qc.X(ancilla);
                break;

            case DJOracle.Balanced:
                // f(x) = x₀ XOR x₁ XOR … (XOR of all input qubits).
                // Oracle: CNOT from each input qubit to ancilla.
                // Phase kick-back: each input qubit individually controls a
                // sign flip on the ancilla |−⟩, encoding parity as a phase.
                for (int i = 0; i < nInput; i++)
                    qc.CNOT(i, ancilla);
                break;
        }

        // Step 4: H on input qubits (NOT ancilla) — causes interference
        for (int i = 0; i < nInput; i++)
            qc.H(i);

        var state = qc.Run();

        // Determine the conclusion.
        // After the algorithm, ALL probability should be at |00…0⟩ (input = 0)
        // for a constant oracle, or zero probability at |00…0⟩ for balanced.
        // The input register occupies qubits 0..nInput-1.
        // Basis state index has the ancilla in the LSB, so input=0 means
        // top nInput bits are 0, i.e. index < 2 (only indices 0 and 1 for ancilla).
        double probAllZeroInput = 0.0;
        for (int i = 0; i < state.NStates; i++)
        {
            // Check if all input qubits (the top nInput bits of i) are 0.
            if ((i >> 1) == 0)
                probAllZeroInput += state.Probabilities()[i];
        }

        string conclusion = probAllZeroInput > 0.9 ? "CONSTANT" : "BALANCED";
        string oracleName = oracle.ToString();

        string desc =
            $"Deutsch-Jozsa  ({nInput} input qubits, oracle = {oracleName})\n" +
            $"  One oracle call distinguishes constant vs balanced f.\n" +
            $"  Result: f is {conclusion}";

        return new AlgorithmResult(qc, state, desc);
    }

    // =========================================================================
    //  GROVER'S SEARCH ALGORITHM
    //
    //  PROBLEM
    //  -------
    //  Find a marked item in an unstructured list of N = 2ⁿ items.
    //  Classical: O(N) queries.  Quantum: O(√N) — quadratic speedup.
    //
    //  GEOMETRIC INTUITION
    //  -------------------
    //  The initial equal superposition |s⟩ is close to the subspace orthogonal
    //  to the target |t⟩.  Each oracle + diffusion step rotates |s⟩ toward |t⟩
    //  by a fixed angle 2θ where sin(θ) = 1/√N.  After ~π√N/4 steps, |s⟩ ≈ |t⟩.
    //
    //  ONE ITERATION CIRCUIT
    //  ─────────────────────
    //  1. Oracle Oₜ: flip phase of |target⟩.
    //     • X gates to flip qubits where target bit = 0
    //     • Multi-controlled Z (CZ / Toffoli-based)
    //     • X gates to undo the flips
    //
    //  2. Diffusion D = 2|s⟩⟨s| - I (inversion about the mean):
    //     H^⊗n → X^⊗n → multi-controlled-Z → X^⊗n → H^⊗n
    // =========================================================================

    /// <summary>
    /// Run Grover's search algorithm.
    /// </summary>
    /// <param name="nQubits">Number of qubits (N = 2^nQubits items).</param>
    /// <param name="target">Integer index of the marked item.</param>
    public static AlgorithmResult Grover(int nQubits = 2, int target = 2)
    {
        int n = nQubits;
        int N = 1 << n;

        if (target < 0 || target >= N)
            throw new ArgumentOutOfRangeException(nameof(target), $"Must be 0–{N-1}.");

        int nIterations = Math.Max(1, (int)Math.Round(Math.PI / 4.0 * Math.Sqrt(N)));

        var qc = new QuantumCircuit(n);

        // Step 1: Uniform superposition
        for (int i = 0; i < n; i++)
            qc.H(i);

        // Decompose target into its bit array (MSB first, qubit 0 = MSB)
        int[] targetBits = new int[n];
        for (int i = 0; i < n; i++)
            targetBits[i] = (target >> (n - 1 - i)) & 1;

        for (int iter = 0; iter < nIterations; iter++)
        {
            // ── Oracle: phase flip of |target⟩ ───────────────────────────
            // Flip qubits where target has a 0-bit so that the target maps to |1…1⟩
            for (int i = 0; i < n; i++)
                if (targetBits[i] == 0) qc.X(i);

            ApplyMultiControlledZ(qc, n);

            // Undo the flips
            for (int i = 0; i < n; i++)
                if (targetBits[i] == 0) qc.X(i);

            // ── Diffusion: 2|s⟩⟨s| - I ───────────────────────────────────
            for (int i = 0; i < n; i++) qc.H(i);
            for (int i = 0; i < n; i++) qc.X(i);

            ApplyMultiControlledZ(qc, n);

            for (int i = 0; i < n; i++) qc.X(i);
            for (int i = 0; i < n; i++) qc.H(i);
        }

        var state = qc.Run();
        double[] probs = state.Probabilities();

        // Find the highest-probability state.
        int found = 0;
        for (int i = 1; i < N; i++)
            if (probs[i] > probs[found]) found = i;

        bool success = found == target;
        string desc =
            $"Grover's Search  ({n} qubits, N={N}, target={target} = " +
            $"|{Convert.ToString(target, 2).PadLeft(n, '0')}⟩, {nIterations} iteration(s))\n" +
            $"  Classical: O({N}) queries.  Quantum: O({(int)Math.Sqrt(N)}) queries.\n" +
            $"  Highest-probability state: |{Convert.ToString(found, 2).PadLeft(n, '0')}⟩ " +
            $"(P={probs[found]:F3})  {(success ? "✓ correct" : "✗ missed")}";

        return new AlgorithmResult(qc, state, desc);
    }

    // -------------------------------------------------------------------------
    //  Helper: add a multi-controlled Z to the circuit.
    //  Works for 1, 2, 3, and 4 qubits.
    // -------------------------------------------------------------------------
    private static void ApplyMultiControlledZ(QuantumCircuit qc, int n)
    {
        switch (n)
        {
            case 1:
                qc.Z(0);
                break;
            case 2:
                // CZ decomposes as: H(q1) · CNOT(0→1) · H(q1)
                qc.H(1);
                qc.CNOT(0, 1);
                qc.H(1);
                break;
            case 3:
                // CCZ: H(q2) · Toffoli(0,1,2) · H(q2)
                qc.H(2);
                qc.Toffoli(0, 1, 2);
                qc.H(2);
                break;
            case 4:
                // 4-qubit: decompose into two Toffoli layers
                qc.H(3);
                qc.Toffoli(0, 1, 2);
                qc.CNOT(2, 3);
                qc.Toffoli(0, 1, 2);
                qc.H(3);
                break;
            default:
                throw new NotSupportedException($"Multi-controlled Z not implemented for {n} qubits.");
        }
    }

    // =========================================================================
    //  QUANTUM TELEPORTATION
    //
    //  Transfers an unknown qubit state |ψ⟩ from Alice to Bob using:
    //    • 1 pre-shared Bell pair (entangled pair)
    //    • 2 classical bits of communication
    //
    //  PROTOCOL
    //  --------
    //  q0 = message (Alice),  q1 = Alice's EPR qubit,  q2 = Bob's EPR qubit
    //
    //  1. Prepare message |ψ⟩ = Ry(π/3)|0⟩ on q0.
    //  2. Create Bell pair (q1, q2): H(q1), CNOT(q1→q2).
    //  3. Alice's Bell-basis rotation: CNOT(q0→q1), H(q0).
    //  4. Alice measures q0, q1 → 2 classical bits b₀, b₁.
    //  5. Bob applies: X to q2 if b₁=1, then Z to q2 if b₀=1.
    //     After corrections q2 is in state |ψ⟩.
    //
    //  NOTE: This method shows the joint state after step 3 (before measurement).
    // =========================================================================

    /// <summary>
    /// Set up the quantum teleportation protocol and return the joint state
    /// after Alice's Bell-basis rotation (before measurement).
    /// </summary>
    public static AlgorithmResult Teleportation()
    {
        var qc = new QuantumCircuit(3);

        // Prepare message qubit: Ry(π/3)|0⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩
        qc.Ry(0, Math.PI / 3.0);

        // Create Bell pair between q1 (Alice) and q2 (Bob)
        qc.H(1);
        qc.CNOT(1, 2);

        // Alice's Bell measurement preparation
        qc.CNOT(0, 1);
        qc.H(0);

        var state = qc.Run();
        string desc =
            "Quantum Teleportation (3 qubits)\n" +
            "  q0 = message: Ry(π/3)|0⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩\n" +
            "  q1 = Alice's entangled qubit,  q2 = Bob's entangled qubit\n" +
            "  State shown: after Alice's Bell-measurement preparation.\n" +
            "  Alice measures q0,q1 → 2 classical bits → Bob applies X/Z to q2.";

        return new AlgorithmResult(qc, state, desc);
    }
}
