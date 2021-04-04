using System;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;

namespace AnomalyDetection
{
    class Program
    {
        private static readonly string BaseModelsRelativePath = @"../../../MLModels";

        private static string SpikePath = GetAbsolutePath($"{BaseModelsRelativePath}/SpikeModel.zip");
        private static string ChangePointPath = GetAbsolutePath($"{BaseModelsRelativePath}/ChangepointModel.zip");

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "blob.csv");

        //assign the Number of records in dataset file to constant variable
        const int _docsize = 36;

        static void Main(string[] args)
        {
            Console.Clear();

            // Create MLContext to be shared across the model creation workflow objects
            MLContext mlContext = new MLContext();

            //STEP 1: Common data loading configuration

            IDataView dataView = mlContext.Data.LoadFromTextFile<AnomalyInput>(path: _dataPath, hasHeader: true, separatorChar: ',');

            // Spike detects pattern temporary changes

            DetectSpike(mlContext, _docsize, dataView);

            // Changepoint detects pattern persistent changes

            DetectChangepoint(mlContext, _docsize, dataView);

            Console.ReadKey();
        }

        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detect temporary changes in pattern");

            // STEP 2: Set the training algorithm

            // SnippetAddSpikeTrainer

            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(AnomalyOutput.Value), inputColumnName: nameof(AnomalyInput.Value), confidence: 95.0, pvalueHistoryLength: docSize / 4);

            // STEP 3: Create the transform
            // Create the spike detection transform

            Console.WriteLine("=============== Training the model ===============");

            // SnippetTrainModel1

            ITransformer trainedModel = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

            Console.WriteLine("=============== End of training process ===============");

            //Apply data transformation to create predictions.

            // SnippetTransformData1

            IDataView transformedData = trainedModel.Transform(productSales);

            // SnippetCreateEnumerable1

            var predictions = mlContext.Data.CreateEnumerable<AnomalyOutput>(transformedData, reuseRowObject: false);

            // Save model in file

            mlContext.Model.Save(trainedModel, productSales.Schema, SpikePath);

            // DisplayHeader1
            Console.WriteLine("Alert\tScore\tP-Value");

            // DisplayResults1

            int position = 0;

            foreach (var p in predictions)
            {
                var results = $"{p.Value[0]}\t{p.Value[1]:f2}\t{p.Value[2]:F2}";

                if (p.Value[0] == 1)
                {
                    results += $" <-- Spike detected at position {position}";
                }

                Console.WriteLine(results);

                position++;
            }
            Console.WriteLine("");
        }

        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detect Persistent changes in pattern");

            //STEP 2: Set the training algorithm

            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(AnomalyOutput.Value), inputColumnName: nameof(AnomalyInput.Value), confidence: 95.0, changeHistoryLength: docSize / 4);

            //STEP 3: Create the transform

            Console.WriteLine("=============== Training the model Using Change Point Detection Algorithm===============");

            // TrainModel2

            ITransformer trainedModel = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

            Console.WriteLine("=============== End of training process ===============");

            //Apply data transformation to create predictions.

            // TransformData2

            IDataView transformedData = trainedModel.Transform(productSales);

            // CreateEnumerable2

            var predictions = mlContext.Data.CreateEnumerable<AnomalyOutput>(transformedData, reuseRowObject: false);

            // DisplayHeader2
            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            mlContext.Model.Save(trainedModel, productSales.Schema, ChangePointPath);

            int position = 0;

            // DisplayResults2
            foreach (var p in predictions)
            {
                var results = $"{p.Value[0]}\t{p.Value[1]:f2}\t{p.Value[2]:F2}\t{p.Value[3]:F2}";

                if (p.Value[0] == 1)
                {
                    results += $" <-- alert is on, predicted changepoint at position {position}";
                }

                Console.WriteLine(results);

                position++;
            }

            Console.WriteLine("");
        }

        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<AnomalyInput> enumerableData = new List<AnomalyInput>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
