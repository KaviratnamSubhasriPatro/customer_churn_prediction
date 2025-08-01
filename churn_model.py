from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_model(csv_path):
    spark = SparkSession.builder \
        .appName("CustomerChurnPrediction") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.withColumn("TotalCharges",
            when(col("TotalCharges") == " ", None).otherwise(col("TotalCharges"))
        ).withColumn("TotalCharges", col("TotalCharges").cast("double")) \
        .dropna(subset=["TotalCharges"])

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    cats = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_Idx", handleInvalid="keep") for c in cats]
    encoders = [OneHotEncoder(inputCol=c+"_Idx", outputCol=c+"_Vec") for c in cats]

    nums = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
    assembler = VectorAssembler(
        inputCols=[c+"_Vec" for c in cats] + nums,
        outputCol="features"
    )

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])

    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    preds_pd = preds.select("label","probability","prediction").toPandas()
    preds_pd["churn_prob"] = preds_pd["probability"].apply(lambda x: float(x[1]))

    evaluator = BinaryClassificationEvaluator()
    auc_val = evaluator.evaluate(preds)

    spark.stop()
    return model, preds_pd, auc_val
