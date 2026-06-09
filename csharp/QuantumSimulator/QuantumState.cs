// =============================================================================
//  FOUR-QUBIT QUANTUM COMPUTING SIMULATOR — C# / .NET 8
//  A fully annotated, educational implementation
// =============================================================================
//
//  BACKGROUND
//  ----------
//  Quantum computers exploit two phenomena with no classical analogue:
//    • Superposition  — a qubit can be BOTH |0⟩ and |1⟩ simultaneously until
//                       measured.
//    • Entanglement   — two or more qubits can share a joint quantum state
//                       such that measuring one instantly determines the other.
//
//  This simulator models both exactly, using the full mathematical formalism:
//  a quantum state of n qubits is a complex vector of length 2ⁿ, and every
//  gate is a unitary matrix.
//
//    n=1 qubit  →  2  amplitudes  (complex vector length 2)
//    n=2 qubits →  4  amplitudes
//    n=3 qubits →  8  amplitudes
//    n=4 qubits → 16  amplitudes
//
//  Each amplitude αᵢ is a complex number.  The probability of measuring basis
//  state |i⟩ is |αᵢ|² and the Born rule requires Σ|αᵢ|² = 1.
//
//  FILE LAYOUT
//  -----------
//    QuantumState.cs  — state vector, measurement, display  (this file)
//    Gates.cs         — all single- and multi-qubit gates + applicators
//    QuantumCircuit.cs— composable gate pipeline
//    Visualiser.cs    — ASCII probability bar chart
//    Algorithms.cs    — Bell, GHZ, Deutsch-Jozsa, Grover, Teleportation
//    Program.cs       — demo runner (entry point)
// =============================================================================

using System.Numerics;
using System.Text;

namespace QuantumSimulator;

// ---------------------------------------------------------------------------
//  Complex arithmetic helpers
//  .NET's System.Numerics.Complex is used throughout.  We add a few
//  convenience extension methods here to keep the gate and state code tidy.
// ---------------------------------------------------------------------------

internal static class ComplexExtensions
{
    /// <summary>Returns |z|² (the squared magnitude, always real and ≥ 0).</summary>
    public static double MagnitudeSquared(this Complex z) =>
        z.Real * z.Real + z.Imaginary * z.Imaginary;

    /// <summary>Formats a complex number as "a + bi" or "a - bi".</summary>
    public static string ToAmplitudeString(this Complex z)
    {
        double re = z.Real;
        double im = z.Imaginary;
        string sign = im >= 0 ? "+" : "-";
        return $"{re:+0.0000;-0.0000} {sign} {Math.Abs(im):0.0000}i";
    }
}

// ---------------------------------------------------------------------------
//  QuantumState
//
//  Represents the full quantum state of an n-qubit system as a state-vector
//  (ket |ψ⟩) in the computational basis.
//
//  MATHEMATICS
//  -----------
//  For n qubits, the state lives in ℂ^(2ⁿ):
//
//      |ψ⟩ = α₀|00…0⟩ + α₁|00…1⟩ + … + α_(2ⁿ-1)|11…1⟩
//
//  where each αᵢ ∈ ℂ and Σᵢ |αᵢ|² = 1  (normalisation).
//
//  Qubit ordering convention (IMPORTANT for multi-qubit gates):
//      Qubit 0 is the MOST SIGNIFICANT BIT (leftmost in ket notation).
//      Index 5 = 0b101 → |101⟩ = qubit0=1, qubit1=0, qubit2=1.
// ---------------------------------------------------------------------------

public sealed class QuantumState
{
    // The amplitude vector.  Index i corresponds to basis state |i⟩.
    public Complex[] Amplitudes { get; private set; }

    /// <summary>Number of qubits.</summary>
    public int NQubits { get; }

    /// <summary>Dimension of the Hilbert space: 2^NQubits.</summary>
    public int NStates { get; }

    private static readonly Random Rng = new(42);  // fixed seed for reproducibility

    // -------------------------------------------------------------------------
    //  Constructors
    // -------------------------------------------------------------------------

    /// <summary>
    /// Initialise in a pure computational basis state.
    /// </summary>
    /// <param name="nQubits">Number of qubits (1–10).</param>
    /// <param name="initialState">
    ///   Integer index of the basis state to start in (default 0 = |0…0⟩).
    /// </param>
    public QuantumState(int nQubits, int initialState = 0)
    {
        if (nQubits < 1 || nQubits > 10)
            throw new ArgumentOutOfRangeException(nameof(nQubits), "Must be 1–10.");

        int nStates = 1 << nQubits;   // 2^nQubits using a left-shift

        if (initialState < 0 || initialState >= nStates)
            throw new ArgumentOutOfRangeException(nameof(initialState),
                $"Must be 0–{nStates - 1}.");

        NQubits = nQubits;
        NStates = nStates;

        // Allocate all-zero amplitudes, then place 1.0 at the chosen basis state.
        // This is a pure state with no superposition.
        Amplitudes = new Complex[NStates];
        Amplitudes[initialState] = Complex.One;
    }

    /// <summary>
    /// Build a QuantumState directly from an amplitude array.
    /// The array length must be a power of 2 and the state must be normalised.
    /// </summary>
    public static QuantumState FromAmplitudes(Complex[] amplitudes)
    {
        int n = amplitudes.Length;
        if (n == 0 || (n & (n - 1)) != 0)
            throw new ArgumentException("Length must be a power of 2.", nameof(amplitudes));

        int nQubits = (int)Math.Log2(n);
        var qs = new QuantumState(nQubits);
        qs.Amplitudes = (Complex[])amplitudes.Clone();

        double norm = qs.Norm();
        if (Math.Abs(norm - 1.0) > 1e-6)
            throw new ArgumentException($"State vector must be normalised (norm = {norm:F6}).");

        return qs;
    }

    // -------------------------------------------------------------------------
    //  State inspection
    // -------------------------------------------------------------------------

    /// <summary>
    /// Return the probability of each computational basis state.
    ///
    /// Born rule: P(i) = |αᵢ|²
    ///
    /// This is the only way to extract classical information from a quantum
    /// state.  The act of measurement disturbs the state (see Measure below).
    /// </summary>
    public double[] Probabilities()
    {
        var probs = new double[NStates];
        for (int i = 0; i < NStates; i++)
            probs[i] = Amplitudes[i].MagnitudeSquared();
        return probs;
    }

    /// <summary>Euclidean norm of the amplitude vector (should always be ≈ 1).</summary>
    public double Norm()
    {
        double sum = 0;
        foreach (var a in Amplitudes)
            sum += a.MagnitudeSquared();
        return Math.Sqrt(sum);
    }

    /// <summary>True if Σ|αᵢ|² ≈ 1 (the state is physically valid).</summary>
    public bool IsNormalised(double atol = 1e-8)
    {
        double sum = 0;
        foreach (var a in Amplitudes)
            sum += a.MagnitudeSquared();
        return Math.Abs(sum - 1.0) < atol;
    }

    /// <summary>Returns the ket label for a basis state index, e.g. index 5 → "|101⟩".</summary>
    public string StateLabel(int index) =>
        $"|{Convert.ToString(index, 2).PadLeft(NQubits, '0')}⟩";

    // -------------------------------------------------------------------------
    //  Measurement — full system
    // -------------------------------------------------------------------------

    /// <summary>
    /// Simulate projective measurement in the computational basis.
    ///
    /// THEORY
    /// ------
    /// Measurement is the irreversible process by which a quantum state
    /// "collapses" into one of the classical basis states.
    ///
    /// Before: |ψ⟩ = Σ αᵢ|i⟩  — all basis states coexist.
    /// After:  system found in state |k⟩ with probability P(k) = |αₖ|².
    ///
    /// The state collapses to |k⟩ — any subsequent measurement gives k.
    ///
    /// This implementation uses a weighted random selection that faithfully
    /// implements the Born rule.
    /// </summary>
    /// <param name="shots">
    ///   Number of independent measurements.  Each shot starts from the
    ///   current (post-collapse) state, so use a fresh QuantumState or call
    ///   Reset() between shots for independent preparations.
    /// </param>
    public int[] Measure(int shots = 1)
    {
        var outcomes = new int[shots];

        for (int shot = 0; shot < shots; shot++)
        {
            double[] probs = Probabilities();

            // Weighted random sampling: walk along the cumulative distribution
            // and find the first index where the running sum exceeds a uniform
            // random number in [0, 1).  This is the inverse-CDF method and
            // exactly implements the Born rule.
            double r = Rng.NextDouble();
            double cumulative = 0.0;
            int outcome = NStates - 1;   // fallback for floating-point rounding
            for (int i = 0; i < NStates; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                {
                    outcome = i;
                    break;
                }
            }

            outcomes[shot] = outcome;

            // State collapse: the system is now definitively in |outcome⟩.
            Array.Clear(Amplitudes, 0, NStates);
            Amplitudes[outcome] = Complex.One;
        }

        return outcomes;
    }

    // -------------------------------------------------------------------------
    //  Measurement — single qubit (partial measurement)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Measure a SINGLE qubit, leaving the rest of the system in a
    /// renormalised conditional state.
    ///
    /// THEORY
    /// ------
    /// 1. Compute marginal P(qubit=0) and P(qubit=1) by summing |αᵢ|²
    ///    over all basis states where the target qubit is 0 or 1.
    /// 2. Sample the outcome.
    /// 3. Zero out all amplitudes inconsistent with the outcome.
    /// 4. Renormalise the surviving amplitudes.
    ///
    /// Qubit 0 is the most significant bit; for n qubits qubit k occupies
    /// bit position (n-1-k) of the basis state index.
    /// </summary>
    public int MeasureQubit(int qubit)
    {
        // Bit position for this qubit in our MSB-first convention.
        int bitPos = NQubits - 1 - qubit;

        // Sum probabilities for all states where this qubit = 1.
        double prob1 = 0.0;
        for (int i = 0; i < NStates; i++)
            if (((i >> bitPos) & 1) == 1)
                prob1 += Amplitudes[i].MagnitudeSquared();

        double prob0 = 1.0 - prob1;

        // Sample outcome.
        int outcome = Rng.NextDouble() < prob0 ? 0 : 1;

        // Zero out amplitudes inconsistent with outcome.
        for (int i = 0; i < NStates; i++)
        {
            int bit = (i >> bitPos) & 1;
            if (bit != outcome)
                Amplitudes[i] = Complex.Zero;
        }

        // Renormalise surviving amplitudes.
        double norm = Norm();
        if (norm > 1e-12)
        {
            double invNorm = 1.0 / norm;
            for (int i = 0; i < NStates; i++)
                Amplitudes[i] *= invNorm;
        }

        return outcome;
    }

    // -------------------------------------------------------------------------
    //  Reset
    // -------------------------------------------------------------------------

    /// <summary>Reset the state to |initialState⟩ without allocating new arrays.</summary>
    public void Reset(int initialState = 0)
    {
        Array.Clear(Amplitudes, 0, NStates);
        Amplitudes[initialState] = Complex.One;
    }

    // -------------------------------------------------------------------------
    //  Display
    // -------------------------------------------------------------------------

    /// <summary>
    /// Return a multi-line string showing each basis state, its amplitude,
    /// probability, and a small inline bar chart.
    /// Zero-amplitude states are omitted to keep the output tidy.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"QuantumState ({NQubits} qubit{(NQubits == 1 ? "" : "s")}, {NStates} basis states)");
        sb.AppendLine(new string('─', 52));

        double[] probs = Probabilities();
        for (int i = 0; i < NStates; i++)
        {
            if (Amplitudes[i].Magnitude < 1e-9) continue;

            string label   = StateLabel(i);
            string ampStr  = Amplitudes[i].ToAmplitudeString();
            double prob    = probs[i];
            string bar     = new string('█', (int)(prob * 20));
            sb.AppendLine($"  {label}  {ampStr}  P={prob:F4}  {bar}");
        }

        sb.AppendLine(new string('─', 52));
        double total = probs.Sum();
        sb.AppendLine($"  Σ probabilities = {total:F8}  " +
                      $"({(IsNormalised() ? "✓ normalised" : "✗ NOT normalised")})");
        return sb.ToString();
    }
}
