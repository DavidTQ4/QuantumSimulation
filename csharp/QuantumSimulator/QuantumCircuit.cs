// =============================================================================
//  QuantumCircuit.cs — composable gate pipeline
// =============================================================================
//
//  A QuantumCircuit records an ordered list of gate operations and executes
//  them on a fresh QuantumState when Run() is called.  The circuit itself
//  holds no quantum state — it is purely a reusable description.
//
//  USAGE
//  -----
//      var qc = new QuantumCircuit(2);
//      qc.H(0);            // Hadamard on qubit 0
//      qc.CNOT(0, 1);      // CNOT with control=0, target=1
//      var state = qc.Run();
//      Console.WriteLine(state);
// =============================================================================

using System.Text;

namespace QuantumSimulator;

// ---------------------------------------------------------------------------
//  Internal record for a single gate operation.
// ---------------------------------------------------------------------------

file sealed record GateOp(string Label, GateMatrix Matrix, int[] Qubits);

// ---------------------------------------------------------------------------
//  QuantumCircuit
// ---------------------------------------------------------------------------

public sealed class QuantumCircuit
{
    private readonly List<GateOp> _ops = new();
    private readonly List<string> _history = new();

    /// <summary>Number of qubits this circuit operates on.</summary>
    public int NQubits { get; }

    /// <summary>Initial basis state (used by Run() if no override is given).</summary>
    public int InitialState { get; }

    public QuantumCircuit(int nQubits, int initialState = 0)
    {
        NQubits      = nQubits;
        InitialState = initialState;
    }

    // -------------------------------------------------------------------------
    //  Internal helper — record a gate
    // -------------------------------------------------------------------------

    private void Add(string label, GateMatrix matrix, params int[] qubits)
    {
        _ops.Add(new GateOp(label, matrix, qubits));
        _history.Add($"{label}({string.Join(",", qubits)})");
    }

    // -------------------------------------------------------------------------
    //  Single-qubit gate methods
    // -------------------------------------------------------------------------

    public void I(int qubit)    => Add("I",     Gates.I(),    qubit);
    public void X(int qubit)    => Add("X",     Gates.X(),    qubit);
    public void Y(int qubit)    => Add("Y",     Gates.Y(),    qubit);
    public void Z(int qubit)    => Add("Z",     Gates.Z(),    qubit);
    public void H(int qubit)    => Add("H",     Gates.H(),    qubit);
    public void S(int qubit)    => Add("S",     Gates.S(),    qubit);
    public void T(int qubit)    => Add("T",     Gates.T(),    qubit);

    public void Rx(int qubit, double theta) =>
        Add($"Rx({theta:F3})", Gates.Rx(theta), qubit);

    public void Ry(int qubit, double theta) =>
        Add($"Ry({theta:F3})", Gates.Ry(theta), qubit);

    public void Rz(int qubit, double theta) =>
        Add($"Rz({theta:F3})", Gates.Rz(theta), qubit);

    public void Phase(int qubit, double phi) =>
        Add($"P({phi:F3})", Gates.Phase(phi), qubit);

    // -------------------------------------------------------------------------
    //  Two-qubit gate methods
    // -------------------------------------------------------------------------

    public void CNOT(int control, int target) =>
        Add("CNOT", Gates.CNOT(), control, target);

    public void CZ(int qubit0, int qubit1) =>
        Add("CZ", Gates.CZ(), qubit0, qubit1);

    public void SWAP(int qubit0, int qubit1) =>
        Add("SWAP", Gates.SWAP(), qubit0, qubit1);

    // -------------------------------------------------------------------------
    //  Three-qubit gate methods
    // -------------------------------------------------------------------------

    public void Toffoli(int c0, int c1, int target) =>
        Add("Toffoli", Gates.Toffoli(), c0, c1, target);

    // -------------------------------------------------------------------------
    //  Execution
    // -------------------------------------------------------------------------

    /// <summary>
    /// Execute all gates in sequence and return the final QuantumState.
    ///
    /// A fresh QuantumState is created at the start of each call so the
    /// circuit is side-effect free and can be reused.
    /// </summary>
    /// <param name="initialState">
    ///   Override the initial basis state (null = use this.InitialState).
    /// </param>
    public QuantumState Run(int? initialState = null)
    {
        var state = new QuantumState(NQubits, initialState ?? InitialState);

        foreach (var op in _ops)
        {
            switch (op.Qubits.Length)
            {
                case 1:
                    Gates.ApplySingle(state, op.Matrix, op.Qubits[0]);
                    break;
                case 2:
                    Gates.ApplyTwo(state, op.Matrix, op.Qubits[0], op.Qubits[1]);
                    break;
                case 3:
                    Gates.ApplyThree(state, op.Matrix, op.Qubits[0], op.Qubits[1], op.Qubits[2]);
                    break;
                default:
                    throw new NotSupportedException(
                        $"No applicator for {op.Qubits.Length}-qubit gates.");
            }
        }

        return state;
    }

    // -------------------------------------------------------------------------
    //  Display
    // -------------------------------------------------------------------------

    /// <summary>Return a one-line diagram of the circuit as a gate sequence.</summary>
    public string Diagram()
    {
        var sb = new StringBuilder();
        sb.Append($"Circuit ({NQubits} qubit{(NQubits == 1 ? "" : "s")}): ");
        sb.Append(_history.Count > 0 ? string.Join(" → ", _history) : "(empty)");
        return sb.ToString();
    }

    public override string ToString() => Diagram();
}
