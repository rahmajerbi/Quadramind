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


##### OUTPUT
# root
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

# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+
# |           DateTime|  RR| SPO2| MAP| SBP| DBP|   HR|  PP|     CO| Class|label|
# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+
# |2020-10-18 15:24:00|35.0| 99.9| 0.0| 0.0| 0.0|106.9| 0.0|    0.0|Normal|    1|
# |2020-10-18 15:25:00|36.4|100.0|87.0|98.9|63.1|107.3|35.8|3841.34|Normal|    0|
# |2020-10-18 15:26:00|35.2|100.0|75.2|97.9|63.0|107.5|34.9|3751.75|Normal|    0|
# |2020-10-18 15:27:00|34.0|100.0|74.8|97.2|62.5|107.0|34.7| 3712.9|Normal|    0|
# |2020-10-18 15:28:00|34.9|100.0|74.0|96.0|62.0|107.0|34.0| 3638.0|Normal|    0|
# |2020-10-18 15:29:00|32.9|100.0|73.5|95.5|61.7|106.8|33.8|3609.84|Normal|    0|
# |2020-10-18 15:30:00|35.3|100.0|73.1|94.8|61.4|106.7|33.4|3563.78|Normal|    0|
# |2020-10-18 15:31:00|31.7|100.0|71.9|93.3|60.5|106.1|32.8|3480.08|Normal|    0|
# |2020-10-18 15:32:00|34.5|100.0|71.6|92.8|60.4|106.5|32.4| 3450.6|Normal|    0|
# |2020-10-18 15:33:00|26.2|100.0|68.5|88.9|57.9|105.2|31.0| 3261.2|Normal|    0|
# |2020-10-18 15:34:00|33.5|100.0|72.0|93.2|61.0|106.3|32.2|3422.86|Normal|    0|
# |2020-10-18 15:35:00|25.8|100.0|71.8|93.0|60.7|105.8|32.3|3417.34|Normal|    0|
# |2020-10-18 15:36:00|29.4| 99.8|71.4|93.1|59.5|103.9|33.6|3491.04|Normal|    0|
# |2020-10-18 15:37:00|19.5|100.0|70.7|92.8|59.6|104.2|33.2|3459.44|Normal|    0|
# |2020-10-18 15:38:00|20.4|100.0|72.3|94.6|61.0|106.8|33.6|3588.48|Normal|    0|
# |2020-10-18 15:39:00|28.9|100.0|73.9|97.1|62.0|106.7|35.1|3745.17|Normal|    0|
# |2020-10-18 15:40:00|21.5|100.0|73.3|96.2|61.3|105.9|34.9|3695.91|Normal|    0|
# |2020-10-18 15:41:00|25.5|100.0|72.9|96.0|61.0|105.6|35.0| 3696.0|Normal|    0|
# |2020-10-18 15:42:00|21.2|100.0|71.8|94.1|60.2|105.1|33.9|3562.89|Normal|    0|
# |2020-10-18 15:43:00|16.0|100.0|69.5|91.2|58.2|104.1|33.0| 3435.3|Normal|    0|
# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+
# only showing top 20 rows

# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+-----------+----------+
# |           DateTime|  RR| SPO2| MAP| SBP| DBP|   HR|  PP|     CO| Class|label|   features|prediction|
# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+-----------+----------+
# |2020-10-18 15:24:00|35.0| 99.9| 0.0| 0.0| 0.0|106.9| 0.0|    0.0|Normal|    1|  (2,[],[])|         1|
# |2020-10-18 15:25:00|36.4|100.0|87.0|98.9|63.1|107.3|35.8|3841.34|Normal|    0|[98.9,63.1]|         0|
# |2020-10-18 15:26:00|35.2|100.0|75.2|97.9|63.0|107.5|34.9|3751.75|Normal|    0|[97.9,63.0]|         0|
# |2020-10-18 15:27:00|34.0|100.0|74.8|97.2|62.5|107.0|34.7| 3712.9|Normal|    0|[97.2,62.5]|         0|
# |2020-10-18 15:28:00|34.9|100.0|74.0|96.0|62.0|107.0|34.0| 3638.0|Normal|    0|[96.0,62.0]|         0|
# |2020-10-18 15:29:00|32.9|100.0|73.5|95.5|61.7|106.8|33.8|3609.84|Normal|    0|[95.5,61.7]|         0|
# |2020-10-18 15:30:00|35.3|100.0|73.1|94.8|61.4|106.7|33.4|3563.78|Normal|    0|[94.8,61.4]|         0|
# |2020-10-18 15:31:00|31.7|100.0|71.9|93.3|60.5|106.1|32.8|3480.08|Normal|    0|[93.3,60.5]|         0|
# |2020-10-18 15:32:00|34.5|100.0|71.6|92.8|60.4|106.5|32.4| 3450.6|Normal|    0|[92.8,60.4]|         0|
# |2020-10-18 15:33:00|26.2|100.0|68.5|88.9|57.9|105.2|31.0| 3261.2|Normal|    0|[88.9,57.9]|         0|
# |2020-10-18 15:34:00|33.5|100.0|72.0|93.2|61.0|106.3|32.2|3422.86|Normal|    0|[93.2,61.0]|         0|
# |2020-10-18 15:35:00|25.8|100.0|71.8|93.0|60.7|105.8|32.3|3417.34|Normal|    0|[93.0,60.7]|         0|
# |2020-10-18 15:36:00|29.4| 99.8|71.4|93.1|59.5|103.9|33.6|3491.04|Normal|    0|[93.1,59.5]|         0|
# |2020-10-18 15:37:00|19.5|100.0|70.7|92.8|59.6|104.2|33.2|3459.44|Normal|    0|[92.8,59.6]|         0|
# |2020-10-18 15:38:00|20.4|100.0|72.3|94.6|61.0|106.8|33.6|3588.48|Normal|    0|[94.6,61.0]|         0|
# |2020-10-18 15:39:00|28.9|100.0|73.9|97.1|62.0|106.7|35.1|3745.17|Normal|    0|[97.1,62.0]|         0|
# |2020-10-18 15:40:00|21.5|100.0|73.3|96.2|61.3|105.9|34.9|3695.91|Normal|    0|[96.2,61.3]|         0|
# |2020-10-18 15:41:00|25.5|100.0|72.9|96.0|61.0|105.6|35.0| 3696.0|Normal|    0|[96.0,61.0]|         0|
# |2020-10-18 15:42:00|21.2|100.0|71.8|94.1|60.2|105.1|33.9|3562.89|Normal|    0|[94.1,60.2]|         0|
# |2020-10-18 15:43:00|16.0|100.0|69.5|91.2|58.2|104.1|33.0| 3435.3|Normal|    0|[91.2,58.2]|         0|
# +-------------------+----+-----+----+----+----+-----+----+-------+------+-----+-----------+----------+
# only showing top 20 rows

# Number of correct predictions: 8819
# Total number of data points: 9013
# Accuracy: 0.98
# root
#  |-- DateTime: string (nullable = true)
#  |-- SBP: double (nullable = true)
#  |-- DBP: double (nullable = true)
#  |-- features: vector (nullable = true)
#  |-- prediction: integer (nullable = false)

# -------------------------------------------
# Batch: 0
# -------------------------------------------
# +-------------------------------+--------------------------------------------------------------------+
# |key                            |value                                                               |
# +-------------------------------+--------------------------------------------------------------------+
# |{"Datetime":"2020-10-24 21:37"}|{"DateTime":"2020-10-24 21:37","SBP":93.0,"DBP":51.2,"prediction":0}|
# |{"Datetime":"2020-10-24 21:38"}|{"DateTime":"2020-10-24 21:38","SBP":88.2,"DBP":49.2,"prediction":0}|
# |{"Datetime":"2020-10-24 21:39"}|{"DateTime":"2020-10-24 21:39","SBP":89.6,"DBP":49.8,"prediction":0}|
# |{"Datetime":"2020-10-24 21:40"}|{"DateTime":"2020-10-24 21:40","SBP":90.8,"DBP":50.6,"prediction":0}|
# |{"Datetime":"2020-10-24 21:41"}|{"DateTime":"2020-10-24 21:41","SBP":91.8,"DBP":50.8,"prediction":0}|
# |{"Datetime":"2020-10-24 21:42"}|{"DateTime":"2020-10-24 21:42","SBP":91.3,"DBP":51.3,"prediction":0}|
# |{"Datetime":"2020-10-24 21:43"}|{"DateTime":"2020-10-24 21:43","SBP":97.5,"DBP":52.8,"prediction":0}|
# |{"Datetime":"2020-10-24 21:44"}|{"DateTime":"2020-10-24 21:44","SBP":93.3,"DBP":49.8,"prediction":0}|
# |{"Datetime":"2020-10-24 21:45"}|{"DateTime":"2020-10-24 21:45","SBP":83.1,"DBP":46.7,"prediction":0}|
# |{"Datetime":"2020-10-24 21:46"}|{"DateTime":"2020-10-24 21:46","SBP":84.1,"DBP":47.2,"prediction":0}|
# |{"Datetime":"2020-10-24 21:47"}|{"DateTime":"2020-10-24 21:47","SBP":86.1,"DBP":48.1,"prediction":0}|
# |{"Datetime":"2020-10-24 21:48"}|{"DateTime":"2020-10-24 21:48","SBP":88.6,"DBP":48.9,"prediction":0}|
# |{"Datetime":"2020-10-24 21:49"}|{"DateTime":"2020-10-24 21:49","SBP":90.6,"DBP":50.1,"prediction":0}|
# |{"Datetime":"2020-10-24 21:50"}|{"DateTime":"2020-10-24 21:50","SBP":89.8,"DBP":49.6,"prediction":0}|
# |{"Datetime":"2020-10-24 21:51"}|{"DateTime":"2020-10-24 21:51","SBP":88.9,"DBP":49.7,"prediction":0}|
# |{"Datetime":"2020-10-24 21:52"}|{"DateTime":"2020-10-24 21:52","SBP":87.3,"DBP":49.2,"prediction":0}|
# |{"Datetime":"2020-10-24 21:53"}|{"DateTime":"2020-10-24 21:53","SBP":89.1,"DBP":50.3,"prediction":0}|
# |{"Datetime":"2020-10-24 21:54"}|{"DateTime":"2020-10-24 21:54","SBP":89.9,"DBP":51.0,"prediction":0}|
# |{"Datetime":"2020-10-24 21:55"}|{"DateTime":"2020-10-24 21:55","SBP":90.6,"DBP":51.3,"prediction":0}|
# |{"Datetime":"2020-10-24 21:56"}|{"DateTime":"2020-10-24 21:56","SBP":90.7,"DBP":51.3,"prediction":0}|
# +-------------------------------+--------------------------------------------------------------------+
# only showing top 20 rows


