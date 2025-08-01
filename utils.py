import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_roc(preds_pd):
    fpr, tpr, _ = roc_curve(preds_pd["label"], preds_pd["churn_prob"])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt.gcf()

def plot_distribution(preds_pd):
    plt.figure(figsize=(7,5))
    sns.histplot(preds_pd, x="churn_prob", hue="label", bins=30, kde=True, palette=["green","red"])
    plt.title("Churn Probability Distribution")
    plt.xlabel("Predicted Probability of Churn")
    plt.ylabel("Frequency")
    plt.grid(True)
    return plt.gcf()

def plot_class_pie(preds_pd):
    plt.figure(figsize=(5,5))
    counts = preds_pd["label"].value_counts().sort_index()
    plt.pie(counts, labels=["No Churn","Churn"], autopct="%1.1f%%", startangle=90)
    plt.title("Class Distribution")
    return plt.gcf()

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
