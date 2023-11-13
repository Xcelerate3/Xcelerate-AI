# Xcelerate AI

⚙️ Sixty years of Moore's Law has provided Xcelerate with the computational power to process trillions of data points. Four decades of the internet (accelerated by the COVID-19 pandemic) have given Xcelerate training data worth trillions in token value. Two decades of mobile internet and cloud computing have placed a supercomputer in the hands of every individual. Decades of technological progress have created the necessary conditions for the ascent of generative artificial intelligence.

🚀 Based on over 10,000 financial models, 50,000+ successful business models, and feedback from 100 million users, Xcelerate has trained intelligent business and financial models. These models empower enterprises and communities to swiftly transition to Web3.

## Code examples

Here is a code snippet for training a Xcelerate model.

```
var dataPath = "sentiment.csv";
var mlContext = new XcelerateMLContext();
var loader = mlContext.Data.CreateTextLoader(new[]
    {
        new TextLoader.Column("SentimentText", DataKind.String, 1),
        new TextLoader.Column("Label", DataKind.Boolean, 0),
    },
    hasHeader: true,
    separatorChar: ',');
var data = loader.Load(dataPath);
var learningPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
        .Append(mlContext.BinaryClassification.Trainers.FastTree());
var model = learningPipeline.Fit(data);
```

Now from the model we can make predictions about our digital assets.

```
var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
var prediction = predictionEngine.Predict();
Console.WriteLine("prediction: " + prediction.Prediction);
```

## License

xcelerate-ai is licensed under the MIT license, and it is free to use commercially.


