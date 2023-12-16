from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'

topic_name = 'Topic_test'
output_topic = 'Topic_KMeans'

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
       
if __name__ == "__main__":
    
    # create Spark session
    spark = SparkSession.builder.master('local').config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    

    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("subscribe", "Topic_test") \
        .option("failOnDataLoss", "false") \
        .option("startingOffsets", "earliest") \
        .load() 


    df.printSchema()

    df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    df = df.withColumn("data", from_json(df.value, BloodPressure.get_schema())).select("data.*")

    df.printSchema()


## TRAINING THE MODEL

# Read your data into a DataFrame
data = spark.read.csv(r"C:\Users\User\Desktop\Quadramind-pyspark (1)\Quadramind-pyspark\Kafka\data\historical_data.csv", header=True, inferSchema=True)

# Threshold values for blood pressure levels
normal_sbp_upper = 120
normal_dbp_upper = 80

# Create a new column 'label' based on blood pressure thresholds and zeros
data = data.withColumn(
    'label',
    when(
        (col('SBP') == 0) & (col('DBP') == 0) & (col('HR') >= 80) ,
        1 # Represents 'Anomaly' for both SBP and DBP being zero
    ).when(
        (col('SBP') < normal_sbp_upper) & (col('DBP') < normal_dbp_upper),
        0  # Represents 'Normal'
    ).otherwise(
        1  # Represents other categories
    )
)

# Show the DataFrame with added 'label' column
data.show()

feature_columns = ['SBP', 'DBP']  

# Assemble features into a single column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(data)

# Initialize K-means model with tunable parameters
k = 2  # Set the number of clusters
max_iter = 100  # Maximum number of iterations
init_mode = "k-means||"  # Initialization mode
init_steps = 15  # Number of steps for k-means|| initialization
tolerance = 0.0001  # Convergence tolerance
distance_measure = "euclidean"  # Distance measure
seed = 123  # Random seed

kmeans = KMeans().setK(k).setMaxIter(max_iter).setInitMode(init_mode).setInitSteps(init_steps) \
    .setTol(tolerance).setDistanceMeasure(distance_measure).setSeed(seed)

# Fit the K-means model
model = kmeans.fit(assembled_data)


# Make predictions
predictions = model.transform(assembled_data)

# Show the predictions
predictions.show()

# Compare -1 labels with 0 predictions and 1 labels with 1 predictions
correct_predictions = predictions.filter(
    ((predictions["label"] == 1) & (predictions["prediction"] == 1)) |
    ((predictions["label"] == 0) & (predictions["prediction"] == 0))
)
predictions.filter(
    ((predictions["label"] == 1) & (predictions["prediction"] == 1)) |
    ((predictions["label"] == 0) & (predictions["prediction"] == 0))
)
# Calculate the count of correct predictions
correct_predictions_count = correct_predictions.count()

# Total number of data points
total_data_count = predictions.count()

# Calculate accuracy
accuracy = correct_predictions_count / total_data_count

print(f"Number of correct predictions: {correct_predictions_count}")
print(f"Total number of data points: {total_data_count}")
print(f"Accuracy: {accuracy:.2f}")


coordinates = predictions.select("SBP", "DBP", "prediction").collect()
y = [row['SBP'] for row in coordinates]
x = [row['DBP'] for row in coordinates]
label_colors = [row['prediction'] for row in coordinates]

# Plot the clusters based on labels
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=label_colors, cmap='viridis')
plt.xlabel("SBP")
plt.ylabel("DBP")
plt.title("K-means Clustering (Colors by Labels)")
plt.colorbar(label='Labels')
plt.show()




## REAL TIME STREAMING ANOMALY DETECTION

# Selecting 'DateTime', 'SBP', and 'DBP' columns from the streaming DataFrame (df)
selected_df = df.select('DateTime', 'SBP', 'DBP')

# Convert string columns to numeric types
selected_df = selected_df.withColumn('SBP', selected_df['SBP'].cast('double'))
selected_df = selected_df.withColumn('DBP', selected_df['DBP'].cast('double'))

# Drop rows with null values in 'SBP' or 'DBP' columns
selected_df = selected_df.dropna(subset=['SBP', 'DBP'])

# Assemble 'SBP' and 'DBP' columns into a single 'features' column
assembler = VectorAssembler(inputCols=['SBP', 'DBP'], outputCol='features')
transformed_df = assembler.transform(selected_df)

# Apply the loaded KMeans model to the transformed DataFrame
streaming_predictions = model.transform(transformed_df)

streaming_predictions.printSchema()


assert type(df) == pyspark.sql.dataframe.DataFrame
row_df = streaming_predictions.select(
        to_json(struct("Datetime")).alias('key'),
        to_json(struct('DateTime',  'SBP', 'DBP', 'prediction')).alias("value") )
    
    # Write final result into console for debugging purpose
query = row_df \
        .writeStream \
        .trigger(processingTime='30 seconds') \
        .outputMode("update") \
        .option("truncate", "false")\
        .format("console") \
        .start()
# Writing to Kafka topic processedBPprediction
query = row_df\
        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
        .writeStream\
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'localhost:9092') \
        .option("topic", output_topic) \
        .option("checkpointLocation", "checkpoints") \
        .start()

query.awaitTermination()
spark.stop()


