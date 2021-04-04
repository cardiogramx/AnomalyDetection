using Microsoft.ML.Data;

namespace AnomalyDetection
{
    public class PredictionInput
    {
        [LoadColumn(0)]
        public string X;

        /// <summary>
        /// Field to detect spikes on
        /// </summary>
        [LoadColumn(1)]
        public float Y;
    }

    public class PredictionOutput
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
