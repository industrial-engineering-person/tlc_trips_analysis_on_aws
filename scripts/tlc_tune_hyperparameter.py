from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row



def main():
    spark = SparkSession \
        .builder \
        .appName("tlc-tune-hyperparameter") \
        .getOrCreate()
        
    
    toy_df = data_load(spark)
    alpha, reg_param = tune_parameter(toy_df)
    load_model(spark, alpha, reg_param)

# 학습 데이터 load
def data_load(spark):
    bucket = "onedayproject"
    data_dir = "s3://" + bucket 

    train_df = spark.read.parquet(f"{data_dir}/tlc_tripsdata/train/")
    toy_df = train_df.sample(False, 0.1, seed=1)

    return toy_df

 # 하이퍼 파라미터 튜닝
def tune_parameter(toy_df):
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

    lr = LinearRegression(
        maxIter=30,
        solver="normal",
        labelCol='total_amount',
        featuresCol='feature_vector'
    )

    cv_stages = stages + [lr]

    cv_pipeline = Pipeline(stages=cv_stages)
    param_grid = ParamGridBuilder()\
                    .addGrid(lr.elasticNetParam, [0.1, 0.2, 0.3, 0.4, 0.5])\
                    .addGrid(lr.regParam, [0.01, 0.02, 0.03, 0.04, 0.05])\
                    .build()

    cross_val = CrossValidator(estimator=cv_pipeline,
                            estimatorParamMaps=param_grid,
                            evaluator=RegressionEvaluator(labelCol="total_amount"),
                            numFolds=5)

    cv_model = cross_val.fit(toy_df)
    alpha = cv_model.bestModel.stages[-1]._java_obj.getElasticNetParam()
    reg_param = cv_model.bestModel.stages[-1]._java_obj.getRegParam()
    
    return alpha, reg_param


# 가장 적합한 정확도를 보이는 파라미터 저장 to csv
def load_model(spark, alpha, reg_param):
    bucket = "onedayproject"
    data_dir = "s3://" + bucket 
    
    hyperparam = [(alpha, reg_param)]
    schema = ['alpha', 'reg_param']

    hyper_df = spark.createDataFrame(data=hyperparam, schema=schema)
    hyper_df.coalesce(1).write.option("header",True).format("csv").mode('overwrite').save(f"{data_dir}/tlc_tripsdata/tune/")


if __name__ == "__main__":
    main()





