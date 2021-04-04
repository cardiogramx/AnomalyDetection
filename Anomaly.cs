using Microsoft.ML.Data;

namespace AnomalyDetection
{
    public class AnomalyInput
    {
        [LoadColumn(0)]
        public string Key;

        /// <summary>
        /// Field to detect spikes on
        /// </summary>
        [LoadColumn(1)]
        public float Value;
    }

    public class AnomalyOutput
    {
        /// <summary>
        /// The prediction vector to hold alert, score, and p-value values
        /// </summary>
        [VectorType(3)]
        public double[] Value { get; set; }
    }
}
