import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.sql.types import *


os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'

topic_name = 'Topic_A'
output_topic = 'processedBPprediction'


class BloodPressure:
    
    def get_schema():
        schema = StructType([
            StructField("DateTime", StringType()),
            StructField("RR", StringType()),
            StructField("SPO2", StringType()),
            StructField("MAP", StringType()),
            StructField("SBP", StringType()),
            StructField("DBP", StringType()),
            StructField("HR", StringType()),
            StructField("PP", StringType()),
            StructField("CO", StringType())
        ])
        return schema
    
    @staticmethod
    def removeDuplicates(df):
        df = df.distinct()
        return df
    
if __name__ == "__main__":
    
    # create Spark session
    spark = SparkSession.builder.master('local').config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    

    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("subscribe", "Topic_A") \
        .option("failOnDataLoss", "false") \
        .option("startingOffsets", "earliest") \
        .load() 


    df.printSchema()
    df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    df = df.withColumn("data", from_json(df.value, BloodPressure.get_schema())).select("data.*")

    df.printSchema()
    # Remove duplicates
    df = BloodPressure.removeDuplicates(df)
    assert type(df) == pyspark.sql.dataframe.DataFrame

    row_df = df.select(
        to_json(struct("Datetime")).alias('key'),
        to_json(struct('RR', 'SPO2', 'MAP', 'SBP', 'DBP', 'HR', 'PP', 'CO', 'DateTime')).alias("value") )
    
    # Write final result into console for debugging purpose
    query = row_df \
        .writeStream \
        .trigger(processingTime='30 seconds') \
        .outputMode("update") \
        .option("truncate", "false")\
        .format("console") \
        .start()
    
    query.awaitTermination()