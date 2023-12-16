from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("KMeansAnomalyDetection").getOrCreate()

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


# Assuming the 'model' variable holds the trained KMeans model
model.write().overwrite().save(r"C:\Users\User\Desktop\Quadramind-pyspark (1)\Quadramind-pyspark\PySpark\kmeans2")
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

