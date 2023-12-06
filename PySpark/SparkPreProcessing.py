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
    
    @staticmethod
    def drop_constant_columns(df):
        result = df.copy()
        for column in df.columns:
            if len(df[column].unique()) == 1:
                result = result.drop(column,axis=1)
        return result
    
    @staticmethod
    def drop_corr_features(df):
        corr_matrix = df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        df.drop(to_drop, axis=1, inplace=True)
        return df
    
    @staticmethod
    def BloodPressureClassification(df):
        df = df.withColumn(
            'BP_level',
            when((col('SBP') < 120) & (col('DBP') < 80), lit('Normal'))
            .when((col('SBP') >= 120) & (col('SBP') < 130) & (col('DBP') < 80), lit('Elevated'))
            .when((col('SBP') >= 130) & (col('SBP') < 140) | ((col('DBP') >= 80) & (col('DBP') < 90)), lit('Stage 1 Hypertension'))
            .when((col('SBP') >= 140) | (col('DBP') >= 90), lit('Stage 2 Hypertension'))
            .otherwise(lit('Hypertensive Crisis'))
        )
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
    # Blood Pressure Categorization
    df = BloodPressure.BloodPressureClassification(df)
    # Drop constant attributes
    df = BloodPressure.drop_constant_columns(df)
    # Drop higher correlated variables
    df = BloodPressure.drop_corr_features(df)

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

    #query.awaitTermination()

    # Writing to Kafka topic processedBPprediction
    query = row_df\
        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
        .writeStream\
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("topic", output_topic) \
        .option("checkpointLocation", "checkpoints") \
        .start().awaitTermination()
    
    print(query.status)

    
## OUTPUT
#     root
#  |-- key: binary (nullable = true)
#  |-- value: binary (nullable = true)
#  |-- topic: string (nullable = true)
#  |-- partition: integer (nullable = true)
#  |-- offset: long (nullable = true)
#  |-- timestamp: timestamp (nullable = true)
#  |-- timestampType: integer (nullable = true)

# root
#  |-- DateTime: string (nullable = true)
#  |-- RR: string (nullable = true)
#  |-- SPO2: string (nullable = true)
#  |-- MAP: string (nullable = true)
#  |-- SBP: string (nullable = true)
#  |-- DBP: string (nullable = true)
#  |-- HR: string (nullable = true)
#  |-- PP: string (nullable = true)
#  |-- CO: string (nullable = true)

# -------------------------------------------
# Batch: 0
# -------------------------------------------
# +---+---------------------------------------------------------------------------------------------------------+
# |key|value                                                                                                    |
# +---+---------------------------------------------------------------------------------------------------------+
# |{} |{"RR":"15.3","SPO2":"100","MAP":"71","SBP":"93.1","DBP":"59.8","HR":"104.2","PP":"33.3","CO":"3469.86"}  |
# |{} |{"RR":"2.1","SPO2":"97.1","MAP":"64.7","SBP":"86.3","DBP":"53.7","HR":"102.1","PP":"32.6","CO":"3328.46"}|
# |{} |{"RR":"14","SPO2":"98","MAP":"75.5","SBP":"105.6","DBP":"59.6","HR":"86.6","PP":"46","CO":"3983.6"}      |
# |{} |{"RR":"29.8","SPO2":"100","MAP":"71.8","SBP":"93.7","DBP":"60.1","HR":"91.3","PP":"33.6","CO":"3067.68"} |
# |{} |{"RR":"25.3","SPO2":"99.9","MAP":"75.4","SBP":"97.4","DBP":"63.5","HR":"95.9","PP":"33.9","CO":"3251.01"}|
# |{} |{"RR":"30.5","SPO2":"100","MAP":"74.1","SBP":"95.9","DBP":"62.1","HR":"92.4","PP":"33.8","CO":"3123.12"} |
# |{} |{"RR":"14","SPO2":"100","MAP":"75.9","SBP":"99.4","DBP":"62.7","HR":"89.3","PP":"36.7","CO":"3277.31"}   |
# |{} |{"RR":"14","SPO2":"99.8","MAP":"71.7","SBP":"94.8","DBP":"58.9","HR":"84.6","PP":"35.9","CO":"3037.14"}  |
# |{} |{"RR":"0","SPO2":"98","MAP":"63.9","SBP":"82.5","DBP":"53.6","HR":"87.2","PP":"28.9","CO":"2520.08"}     |
# |{} |{"RR":"14","SPO2":"100","MAP":"63.5","SBP":"83.4","DBP":"52.2","HR":"83.4","PP":"31.2","CO":"2602.08"}   |
# |{} |{"RR":"14","SPO2":"100","MAP":"73.4","SBP":"101.3","DBP":"59.5","HR":"81.4","PP":"41.8","CO":"3402.52"}  |
# |{} |{"RR":"11.2","SPO2":"100","MAP":"77.8","SBP":"105.7","DBP":"63.2","HR":"79.9","PP":"42.5","CO":"3395.75"}|
# |{} |{"RR":"30.1","SPO2":"99.5","MAP":"72.6","SBP":"101.2","DBP":"59","HR":"88.4","PP":"42.2","CO":"3730.48"} |
# |{} |{"RR":"27.7","SPO2":"99.9","MAP":"69.9","SBP":"98.9","DBP":"56.7","HR":"86.7","PP":"42.2","CO":"3658.74"}|
# |{} |{"RR":"12.7","SPO2":"98","MAP":"60.6","SBP":"81.6","DBP":"49.5","HR":"86.5","PP":"32.1","CO":"2776.65"}  |
# |{} |{"RR":"23.8","SPO2":"100","MAP":"65.7","SBP":"92.5","DBP":"52","HR":"77.6","PP":"40.5","CO":"3142.8"}    |
# |{} |{"RR":"19.4","SPO2":"100","MAP":"69.8","SBP":"109.2","DBP":"60.2","HR":"81.7","PP":"49","CO":"4003.3"}   |
# |{} |{"RR":"32.2","SPO2":"99","MAP":"74.5","SBP":"107.3","DBP":"57.6","HR":"82.9","PP":"49.7","CO":"4120.13"} |
# |{} |{"RR":"16.3","SPO2":"100","MAP":"77.7","SBP":"110.7","DBP":"61.2","HR":"79.8","PP":"49.5","CO":"3950.1"} |
# |{} |{"RR":"30","SPO2":"100","MAP":"85.1","SBP":"123.9","DBP":"67.3","HR":"80.9","PP":"56.6","CO":"4578.94"}  |
# +---+---------------------------------------------------------------------------------------------------------+
# only showing top 20 rows

