from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
sc.install_pypi_package("pandas==0.25.1")
import pandas as pd


def main():
    spark = SparkSession \
        .builder \
        .appName("tlc-train-model") \
        .getOrCreate()
        
    model, vtest_df = data_parameter_load_train(spark)
    
    predict(model, vtest_df)

def data_parameter_load_train(spark):
    bucket = "onedayproject"
    data_dir = "s3://" + bucket 

    train_df = spark.read.parquet(f"{data_dir}/tlc_tripsdata/train/")
    test_df = spark.read.parquet(f"{data_dir}/tlc_tripsdata/test/")

    hyper_df_spark = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(f"{data_dir}/tlc_tripsdata/tune/")
    hyper_df = hyper_df_spark.toPandas()

    alpha = float(hyper_df.iloc[0]['alpha'])
    reg_param = float(hyper_df.iloc[0]['reg_param'])


    cat_feats = [
        "pickup_location_id",
        "dropoff_location_id",
        "day_of_week"
    ]

    stages = []

    for c in cat_feats:
        cat_indexer = StringIndexer(inputCol=c, outputCol= c + "_idx").setHandleInvalid("keep")
        onehot_encoder = OneHotEncoder(inputCol=cat_indexer.getOutputCol(), outputCol=c + "_onehot")
        stages += [cat_indexer, onehot_encoder]

    num_feats = [
        "passenger_count",
        "trip_distance",
        "pickup_time"
    ]

    for n in num_feats:
        num_assembler = VectorAssembler(inputCols=[n], outputCol= n + "_vecotr")
        num_scaler = StandardScaler(inputCol=num_assembler.getOutputCol(), outputCol= n + "_scaled")
        stages += [num_assembler, num_scaler]

    assembler_inputs = [c + "_onehot" for c in cat_feats] + [n + "_scaled" for n in num_feats]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="feature_vector")
    stages += [assembler]


    # Training
    transform_stages = stages
    pipeline = Pipeline(stages=transform_stages)
    fitted_transformer = pipeline.fit(train_df)

    vtrain_df = fitted_transformer.transform(train_df)
    lr = LinearRegression(
        maxIter=50,
        solver="normal",
        labelCol="total_amount",
        featuresCol="feature_vector",
        elasticNetParam=alpha,
        regParam=reg_param,
    )

    model = lr.fit(vtrain_df)
    vtest_df = fitted_transformer.transform(test_df)

    return model, vtest_df

# 예측 및 차후 모델을 다시 사용하기 위해 저장
def predict(model, vtest_df):
    predictions = model.transform(vtest_df)
    predictions.cache()
    predictions.select(["trip_distance", "day_of_week", "total_amount", "prediction"]).show()

    model_dir = "s3://onedayproject/tlc_tripsdata/tune/model"

    
    model.write().overwrite().save(model_dir)



if __name__ == "__main__":
    main()
