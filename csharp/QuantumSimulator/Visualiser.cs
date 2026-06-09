// =============================================================================
//  Visualiser.cs — ASCII probability bar chart
// =============================================================================
//
//  Renders the probability distribution of a QuantumState as a bar chart.
//  Works in any terminal — no external GUI dependency.
// =============================================================================

using System.Text;

namespace QuantumSimulator;

public static class Visualiser
{
    private const int BarWidth = 40;   // characters in a full bar (probability = 1.0)

    /// <summary>
    /// Return an ASCII probability histogram as a string.
    ///
    /// Example output for a 2-qubit Bell state:
    ///     |00⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
    ///     |01⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
    ///     |10⟩  ████████████████████░░░░░░░░░░░░░░░░░░░░   50.00%
    ///     |11⟩  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.00%
    /// </summary>
    public static string AsciiBar(QuantumState state, string title = "")
    {
        var sb = new StringBuilder();

        if (!string.IsNullOrEmpty(title))
        {
            string rule = new('═', BarWidth + 20);
            sb.AppendLine();
            sb.AppendLine(rule);
            sb.AppendLine($"  {title}");
            sb.AppendLine(rule);
        }

        double[] probs = state.Probabilities();
        for (int i = 0; i < state.NStates; i++)
        {
            double prob   = probs[i];
            string label  = state.StateLabel(i);
            int filled    = (int)Math.Round(prob * BarWidth);
            string bar    = new string('█', filled) + new string('░', BarWidth - filled);
            sb.AppendLine($"  {label}  {bar}  {prob * 100,6:F2}%");
        }

        sb.AppendLine();
        return sb.ToString();
    }

    /// <summary>
    /// Print an ASCII probability histogram directly to the console.
    /// </summary>
    public static void Print(QuantumState state, string title = "") =>
        Console.Write(AsciiBar(state, title));
}
