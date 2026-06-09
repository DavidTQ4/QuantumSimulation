// =============================================================================
//  Gates.cs — Quantum gate matrices + tensor-contraction applicators
// =============================================================================
//
//  THEORY — WHAT IS A QUANTUM GATE?
//  ----------------------------------
//  A quantum gate is a UNITARY matrix U acting on the state vector:
//      |ψ'⟩ = U|ψ⟩
//
//  Unitary means  U†U = I  (U† = conjugate transpose).
//  This guarantees:
//    • Normalisation is preserved (gates never violate probability).
//    • The operation is reversible (U⁻¹ = U†).
//
//  For a single qubit the gate is 2×2; for n qubits the full-system gate is
//  2ⁿ × 2ⁿ.  When a gate acts on only one qubit we embed it via tensor product:
//      U_full = I ⊗ … ⊗ U ⊗ … ⊗ I
//
//  Instead of building this huge matrix explicitly, we use a tensor
//  contraction over the reshaped amplitude array (see ApplySingle below).
//
//  ALL MATRICES use the computational basis {|0⟩, |1⟩}:
//    row/column 0 = |0⟩,  row/column 1 = |1⟩.
//
//  For two-qubit matrices the four-element basis is ordered:
//    {|00⟩, |01⟩, |10⟩, |11⟩} i.e. qubit 0 is MSB.
// =============================================================================

using System.Numerics;

namespace QuantumSimulator;

/// <summary>
/// Represents a gate matrix together with its qubit arity,
/// purely for documentation / type-safety when passing gates around.
/// The underlying data is a 1-D row-major array of size (2^arity)².
/// </summary>
public readonly struct GateMatrix
{
    public readonly Complex[] Data;   // row-major, size = dim*dim
    public readonly int Dim;          // 2^arity  (2 for 1-qubit, 4 for 2-qubit, …)
    public readonly int Arity;        // number of qubits the gate acts on

    public GateMatrix(Complex[] data, int arity)
    {
        Arity = arity;
        Dim   = 1 << arity;
        if (data.Length != Dim * Dim)
            throw new ArgumentException($"Gate data length {data.Length} != {Dim * Dim}");
        Data = data;
    }

    /// <summary>Row-major element access: element [row, col].</summary>
    public Complex this[int row, int col] => Data[row * Dim + col];
}

// ---------------------------------------------------------------------------
//  Gates — factory methods for every standard gate
// ---------------------------------------------------------------------------

public static class Gates
{
    // Convenient shorthand for constructing Complex values.
    private static Complex C(double re, double im = 0) => new(re, im);

    private const double Sqrt2Inv = 0.7071067811865476;   // 1/√2

    // =========================================================================
    //  SINGLE-QUBIT GATES
    // =========================================================================

    /// <summary>
    /// Identity gate — does nothing.
    ///
    ///     I = [ 1  0 ]
    ///         [ 0  1 ]
    ///
    /// Used as a placeholder in tensor-product constructions.
    /// </summary>
    public static GateMatrix I() => new(new Complex[]
    {
        C(1), C(0),
        C(0), C(1)
    }, arity: 1);

    /// <summary>
    /// Pauli-X gate — quantum NOT / bit-flip.
    ///
    ///     X = [ 0  1 ]
    ///         [ 1  0 ]
    ///
    ///     X|0⟩ = |1⟩     X|1⟩ = |0⟩
    ///
    /// The exact quantum analogue of a classical NOT gate.
    /// On a superposition X(α|0⟩ + β|1⟩) = α|1⟩ + β|0⟩.
    /// </summary>
    public static GateMatrix X() => new(new Complex[]
    {
        C(0), C(1),
        C(1), C(0)
    }, arity: 1);

    /// <summary>
    /// Pauli-Y gate.
    ///
    ///     Y = [  0  -i ]
    ///         [  i   0 ]
    ///
    ///     Y|0⟩ =  i|1⟩     Y|1⟩ = -i|0⟩
    ///
    /// Combines a bit-flip with a phase change.
    /// Together with X and Z forms the Pauli group, which generates all
    /// single-qubit unitaries.
    /// </summary>
    public static GateMatrix Y() => new(new Complex[]
    {
        C(0),  C(0,-1),
        C(0,1), C(0)
    }, arity: 1);

    /// <summary>
    /// Pauli-Z gate — phase flip.
    ///
    ///     Z = [ 1   0 ]
    ///         [ 0  -1 ]
    ///
    ///     Z|0⟩ =  |0⟩    (unchanged)
    ///     Z|1⟩ = -|1⟩    (sign of |1⟩ amplitude flipped)
    ///
    /// The phase flip is INVISIBLE to direct measurement of a single qubit
    /// (|−α|² = |α|²) but determines interference behaviour under further gates.
    /// </summary>
    public static GateMatrix Z() => new(new Complex[]
    {
        C(1), C(0),
        C(0), C(-1)
    }, arity: 1);

    /// <summary>
    /// Hadamard gate — creates and destroys superposition.
    ///
    ///     H = (1/√2) × [ 1   1 ]
    ///                   [ 1  -1 ]
    ///
    ///     H|0⟩ = (|0⟩ + |1⟩)/√2  = |+⟩   (equal superposition)
    ///     H|1⟩ = (|0⟩ - |1⟩)/√2  = |−⟩   (equal superposition with phase)
    ///
    /// Applied twice: H² = I (H is its own inverse).
    ///
    /// Applied to all n qubits of |0…0⟩, H⊗ⁿ creates a uniform superposition
    /// over all 2ⁿ basis states — the standard initialisation for search algorithms.
    /// </summary>
    public static GateMatrix H() => new(new Complex[]
    {
        C(Sqrt2Inv),  C(Sqrt2Inv),
        C(Sqrt2Inv),  C(-Sqrt2Inv)
    }, arity: 1);

    /// <summary>
    /// S gate (phase gate / √Z).
    ///
    ///     S = [ 1  0 ]
    ///         [ 0  i ]
    ///
    ///     S|0⟩ = |0⟩       S|1⟩ = i|1⟩   (adds phase of π/2)
    ///
    /// S² = Z.  Used in the quantum Fourier transform.
    /// </summary>
    public static GateMatrix S() => new(new Complex[]
    {
        C(1), C(0),
        C(0), C(0,1)
    }, arity: 1);

    /// <summary>
    /// T gate (π/8 gate / ⁴√Z).
    ///
    ///     T = [ 1          0        ]
    ///         [ 0   exp(iπ/4)       ]
    ///       = [ 1          0        ]
    ///         [ 0   (1+i)/√2        ]
    ///
    ///     T|0⟩ = |0⟩       T|1⟩ = e^(iπ/4)|1⟩   (adds phase of π/4)
    ///
    /// T⁴ = Z,  T² = S.
    ///
    /// WHY T IS IMPORTANT
    /// ------------------
    /// {H, T} is a UNIVERSAL gate set (Solovay–Kitaev theorem): any unitary on
    /// any number of qubits can be approximated to arbitrary precision using only
    /// these two gates.  T is the "non-Clifford" gate that provides the extra
    /// computational power beyond the Clifford group {H, S, CNOT}.
    /// </summary>
    public static GateMatrix T()
    {
        // e^(iπ/4) = cos(π/4) + i·sin(π/4) = (1+i)/√2
        double angle = Math.PI / 4.0;
        return new GateMatrix(new Complex[]
        {
            C(1), C(0),
            C(0), new Complex(Math.Cos(angle), Math.Sin(angle))
        }, arity: 1);
    }

    /// <summary>
    /// Rotation about the X-axis of the Bloch sphere by angle theta.
    ///
    ///     Rx(θ) = exp(-iθX/2)
    ///           = [ cos(θ/2)    -i·sin(θ/2) ]
    ///             [ -i·sin(θ/2)  cos(θ/2)   ]
    ///
    /// At θ = π this equals -iX (equivalent to X up to global phase).
    /// </summary>
    public static GateMatrix Rx(double theta)
    {
        double c = Math.Cos(theta / 2);
        double s = Math.Sin(theta / 2);
        return new GateMatrix(new Complex[]
        {
            C(c),       C(0,-s),
            C(0,-s),    C(c)
        }, arity: 1);
    }

    /// <summary>
    /// Rotation about the Y-axis of the Bloch sphere by angle theta.
    ///
    ///     Ry(θ) = [ cos(θ/2)  -sin(θ/2) ]
    ///             [ sin(θ/2)   cos(θ/2)  ]
    /// </summary>
    public static GateMatrix Ry(double theta)
    {
        double c = Math.Cos(theta / 2);
        double s = Math.Sin(theta / 2);
        return new GateMatrix(new Complex[]
        {
            C(c),  C(-s),
            C(s),  C(c)
        }, arity: 1);
    }

    /// <summary>
    /// Rotation about the Z-axis of the Bloch sphere by angle theta.
    ///
    ///     Rz(θ) = [ exp(-iθ/2)     0       ]
    ///             [     0       exp(+iθ/2)  ]
    /// </summary>
    public static GateMatrix Rz(double theta)
    {
        return new GateMatrix(new Complex[]
        {
            new Complex(Math.Cos(theta/2), -Math.Sin(theta/2)),  C(0),
            C(0),  new Complex(Math.Cos(theta/2),  Math.Sin(theta/2))
        }, arity: 1);
    }

    /// <summary>
    /// General phase gate — adds phase e^(iφ) to the |1⟩ amplitude.
    ///
    ///     P(φ) = [ 1       0     ]
    ///            [ 0   e^(iφ)    ]
    ///
    /// Special cases:  P(π) = Z,  P(π/2) = S,  P(π/4) = T.
    /// </summary>
    public static GateMatrix Phase(double phi) => new(new Complex[]
    {
        C(1), C(0),
        C(0), new Complex(Math.Cos(phi), Math.Sin(phi))
    }, arity: 1);

    // =========================================================================
    //  TWO-QUBIT GATES
    //  Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩  (qubit 0 = MSB)
    // =========================================================================

    /// <summary>
    /// CNOT (Controlled-NOT / CX) gate — THE entangling gate.
    ///
    /// Convention: qubit 0 = control, qubit 1 = target.
    ///
    ///     CNOT = [ 1  0  0  0 ]
    ///            [ 0  1  0  0 ]
    ///            [ 0  0  0  1 ]
    ///            [ 0  0  1  0 ]
    ///
    ///     |00⟩ → |00⟩    |01⟩ → |01⟩
    ///     |10⟩ → |11⟩    |11⟩ → |10⟩   (target flipped when control = 1)
    ///
    /// WHY CNOT CREATES ENTANGLEMENT
    /// ------------------------------
    /// Start: H|0⟩ ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
    /// After CNOT: (|00⟩ + |11⟩)/√2  ← Bell state — cannot be written as a
    /// product of separate qubit states, so the qubits are entangled.
    /// </summary>
    public static GateMatrix CNOT() => new(new Complex[]
    {
        C(1),C(0),C(0),C(0),
        C(0),C(1),C(0),C(0),
        C(0),C(0),C(0),C(1),
        C(0),C(0),C(1),C(0)
    }, arity: 2);

    /// <summary>
    /// Controlled-Z (CZ) gate.
    ///
    ///     CZ = [ 1  0  0   0 ]
    ///          [ 0  1  0   0 ]
    ///          [ 0  0  1   0 ]
    ///          [ 0  0  0  -1 ]
    ///
    ///     |11⟩ → -|11⟩   (phase flip only when both qubits are |1⟩)
    ///
    /// CZ is symmetric: both qubits play equal roles.
    /// </summary>
    public static GateMatrix CZ() => new(new Complex[]
    {
        C(1),C(0),C(0), C(0),
        C(0),C(1),C(0), C(0),
        C(0),C(0),C(1), C(0),
        C(0),C(0),C(0), C(-1)
    }, arity: 2);

    /// <summary>
    /// SWAP gate — exchanges the states of two qubits.
    ///
    ///     SWAP = [ 1  0  0  0 ]
    ///            [ 0  0  1  0 ]
    ///            [ 0  1  0  0 ]
    ///            [ 0  0  0  1 ]
    ///
    ///     |01⟩ ↔ |10⟩
    ///
    /// Decomposes into three CNOTs: CNOT(0→1) · CNOT(1→0) · CNOT(0→1).
    /// </summary>
    public static GateMatrix SWAP() => new(new Complex[]
    {
        C(1),C(0),C(0),C(0),
        C(0),C(0),C(1),C(0),
        C(0),C(1),C(0),C(0),
        C(0),C(0),C(0),C(1)
    }, arity: 2);

    // =========================================================================
    //  THREE-QUBIT GATES
    // =========================================================================

    /// <summary>
    /// Toffoli gate (CCNOT — Controlled-Controlled-NOT).
    ///
    /// An 8×8 unitary (3 qubits → 8 basis states).
    /// Flips qubit 2 (target) only when BOTH qubit 0 and qubit 1 (controls)
    /// are |1⟩.
    ///
    ///     |110⟩ → |111⟩    |111⟩ → |110⟩
    ///     All other basis states unchanged.
    ///
    /// The Toffoli gate is the quantum analogue of the classical AND gate
    /// (in its reversible, quantum-compatible form).  It is UNIVERSAL for
    /// classical reversible computation.
    /// </summary>
    public static GateMatrix Toffoli()
    {
        // Start from the 8×8 identity, then swap rows/cols 6 (|110⟩) and 7 (|111⟩).
        var data = new Complex[64];
        for (int i = 0; i < 8; i++)
            data[i * 8 + i] = Complex.One;  // identity diagonal

        // Swap the 2×2 block at rows/cols 6,7
        data[6 * 8 + 6] = Complex.Zero;   data[6 * 8 + 7] = Complex.One;
        data[7 * 8 + 6] = Complex.One;    data[7 * 8 + 7] = Complex.Zero;

        return new GateMatrix(data, arity: 3);
    }

    // =========================================================================
    //  GATE APPLICATION ENGINE
    //
    //  Rather than building the full 2ⁿ×2ⁿ unitary via tensor products
    //  (which would be exponentially large), we reshape the amplitude vector
    //  into a multi-dimensional tensor and contract along the target axes.
    //
    //  For a single-qubit gate on qubit k of an n-qubit system:
    //    1. View the amplitude array as an n-dimensional tensor T[b₀,b₁,…,bₙ₋₁]
    //       where each index bᵢ ∈ {0,1} and the flat index is b₀·2^(n-1) + b₁·2^(n-2) + …
    //    2. Contract gate[out_k, in_k] with T[…, in_k, …] over the in_k axis.
    //    3. The result is a new tensor with the same shape, with qubit k updated.
    //
    //  This is O(2ⁿ) in memory and time, not O(4ⁿ).
    // =========================================================================

    /// <summary>
    /// Apply a 2×2 single-qubit gate to one qubit of an n-qubit state.
    ///
    /// For each pair of basis states that differ ONLY in the value of the
    /// target qubit, we apply the 2×2 matrix to the pair of amplitudes.
    /// This is the direct implementation of the tensor contraction for 1 qubit.
    /// </summary>
    public static void ApplySingle(QuantumState state, GateMatrix gate, int qubit)
    {
        int n       = state.NQubits;
        int nStates = state.NStates;

        // Bit position for this qubit in the flat index (MSB-first convention).
        int bitPos = n - 1 - qubit;
        int stride = 1 << bitPos;   // distance between the |…0…⟩ and |…1…⟩ partners

        // Iterate over all pairs of basis states that differ only at `qubit`.
        // For each outer index (with the target qubit's bit set to 0), the
        // partner index is obtained by setting that bit to 1.
        for (int outer = 0; outer < nStates; outer++)
        {
            // Process only the states where qubit = 0 (avoid double-processing).
            if (((outer >> bitPos) & 1) != 0) continue;

            int i0 = outer;           // index with qubit = 0
            int i1 = outer | stride;  // index with qubit = 1

            Complex a0 = state.Amplitudes[i0];
            Complex a1 = state.Amplitudes[i1];

            // Apply the 2×2 gate matrix:
            //   new_a0 = gate[0,0]*a0 + gate[0,1]*a1
            //   new_a1 = gate[1,0]*a0 + gate[1,1]*a1
            state.Amplitudes[i0] = gate[0, 0] * a0 + gate[0, 1] * a1;
            state.Amplitudes[i1] = gate[1, 0] * a0 + gate[1, 1] * a1;
        }
    }

    /// <summary>
    /// Apply a 4×4 two-qubit gate to two qubits of an n-qubit state.
    ///
    /// For each group of four basis states that differ only in the values of
    /// the two target qubits, we apply the 4×4 matrix to those four amplitudes.
    ///
    /// This is mathematically equivalent to the tensor contraction for 2 qubits
    /// but implemented directly without reshaping (to keep C# array indexing simple).
    /// </summary>
    public static void ApplyTwo(QuantumState state, GateMatrix gate,
                                 int qubit0, int qubit1)
    {
        int n       = state.NQubits;
        int nStates = state.NStates;

        int bitPos0 = n - 1 - qubit0;
        int bitPos1 = n - 1 - qubit1;
        int stride0 = 1 << bitPos0;
        int stride1 = 1 << bitPos1;

        for (int outer = 0; outer < nStates; outer++)
        {
            // Process only states where BOTH target qubits are 0.
            if (((outer >> bitPos0) & 1) != 0) continue;
            if (((outer >> bitPos1) & 1) != 0) continue;

            // The four basis state indices for this group.
            int i00 = outer;
            int i01 = outer | stride1;
            int i10 = outer | stride0;
            int i11 = outer | stride0 | stride1;

            Complex a00 = state.Amplitudes[i00];
            Complex a01 = state.Amplitudes[i01];
            Complex a10 = state.Amplitudes[i10];
            Complex a11 = state.Amplitudes[i11];

            // Apply the 4×4 gate (basis order: |q0q1⟩ = |00⟩,|01⟩,|10⟩,|11⟩):
            state.Amplitudes[i00] = gate[0,0]*a00 + gate[0,1]*a01 + gate[0,2]*a10 + gate[0,3]*a11;
            state.Amplitudes[i01] = gate[1,0]*a00 + gate[1,1]*a01 + gate[1,2]*a10 + gate[1,3]*a11;
            state.Amplitudes[i10] = gate[2,0]*a00 + gate[2,1]*a01 + gate[2,2]*a10 + gate[2,3]*a11;
            state.Amplitudes[i11] = gate[3,0]*a00 + gate[3,1]*a01 + gate[3,2]*a10 + gate[3,3]*a11;
        }
    }

    /// <summary>
    /// Apply an 8×8 three-qubit gate (e.g. Toffoli) to three qubits of an n-qubit state.
    ///
    /// Same strategy as ApplyTwo: iterate over groups of 8 basis states
    /// that share the same values for all qubits OTHER than the three targets,
    /// and apply the 8×8 matrix to each group.
    /// </summary>
    public static void ApplyThree(QuantumState state, GateMatrix gate,
                                   int qubit0, int qubit1, int qubit2)
    {
        int n       = state.NQubits;
        int nStates = state.NStates;

        int bitPos0 = n - 1 - qubit0;
        int bitPos1 = n - 1 - qubit1;
        int bitPos2 = n - 1 - qubit2;
        int stride0 = 1 << bitPos0;
        int stride1 = 1 << bitPos1;
        int stride2 = 1 << bitPos2;

        // We need a temporary buffer to hold the 8 old amplitudes per group.
        var old = new Complex[8];

        for (int outer = 0; outer < nStates; outer++)
        {
            if (((outer >> bitPos0) & 1) != 0) continue;
            if (((outer >> bitPos1) & 1) != 0) continue;
            if (((outer >> bitPos2) & 1) != 0) continue;

            // Compute the eight indices for this group.
            Span<int> idx = stackalloc int[8];
            idx[0] = outer;
            idx[1] = outer | stride2;
            idx[2] = outer | stride1;
            idx[3] = outer | stride1 | stride2;
            idx[4] = outer | stride0;
            idx[5] = outer | stride0 | stride2;
            idx[6] = outer | stride0 | stride1;
            idx[7] = outer | stride0 | stride1 | stride2;

            for (int k = 0; k < 8; k++)
                old[k] = state.Amplitudes[idx[k]];

            for (int row = 0; row < 8; row++)
            {
                Complex sum = Complex.Zero;
                for (int col = 0; col < 8; col++)
                    sum += gate[row, col] * old[col];
                state.Amplitudes[idx[row]] = sum;
            }
        }
    }
}
