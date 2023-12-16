import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.pipeline import PipelineModel

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = "C:\\hadoop"

topic_name = 'Topic_test'
output_topic = 'Topic_KMeans'

def predict_anomaly(row):
    sbp = float(row['SBP'])
    dbp = float(row['DBP'])
    features = [sbp, dbp]  # Modify this as per your feature processing
    prediction = loaded_model.predict([features])
    return str(prediction)

if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder.master('local').config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0").config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    
   

    # Read the pickled model via the Pipeline API
    loaded_model = KMeansModel.load(r"C:\Users\User\Desktop\Quadramind-pyspark (1)\Quadramind-pyspark\PySpark\kmeans2")

    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("subscribe", "Topic_test") \
        .option("failOnDataLoss", "false") \
        .option("startingOffsets", "earliest") \
        .load() 

    assert type(df) == pyspark.sql.dataframe.DataFrame
    row_df = df.select(
        to_json(struct("Datetime")).alias('key'),
        to_json(struct('DateTime', 'RR', 'SPO2', 'MAP', 'SBP', 'DBP', 'HR', 'PP', 'CO')).alias("value") 
    )
    
    predict_udf = udf(predict_anomaly, StringType())
    detected_anomalies = row_df.withColumn("anomaly_prediction", predict_udf(struct(*row_df.columns)))
    
    query = detected_anomalies \
        .writeStream \
        .trigger(processingTime='30 seconds') \
        .outputMode("update") \
        .option("truncate", "false")\
        .format("console") \
        .start()

    query = detected_anomalies \
        .selectExpr("CAST(key AS STRING)", "CAST(anomaly_prediction AS STRING)") \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("topic", output_topic) \
        .option("checkpointLocation", "checkpoints") \
        .start().awaitTermination()

    print(query.status)
