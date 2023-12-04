from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, MinMaxScaler


# Create a SparkSession
spark = SparkSession.builder.appName("BPprocessing").getOrCreate()

# Read a CSV file into a DataFrame
df = spark.read.csv("./Kafka/mimic_2425.csv", header=True, inferSchema=True)
df.show(5)

# Number of rows
row_count = df.count()
print("Number of rows in the DataFrame:", row_count)

# Display summary statistics for numerical columns in the DataFrame
df.describe().show()

# Iterate through the DataFrame schema and print column names along with their data types
for col_name, data_type in df.dtypes:
    print(f"Column '{col_name}' has data type '{data_type}'")


# Get column names except the first one (it's a DateTime column)
columns_to_check = df.columns[1:]

# Check for missing values in the DataFrame for columns after the DateTime column
missing_count = df.select([col(c).isNull().alias(c) for c in columns_to_check])

# Count missing values for each column
missing_values_per_column = missing_count.agg(*[count(when(col(c), c)).alias(c) for c in columns_to_check])

# Display missing value counts for each column
print("Missing values for each column: \n")
missing_values_per_column.show()

# Select numerical columns for correlation
numeric_cols = ['RR', 'SPO2', 'MAP', 'SBP', 'DBP', 'HR', 'PP', 'CO']

# Convert selected columns to a single vector column
assembler = VectorAssembler(inputCols=numeric_cols, outputCol='features')
vector_df = assembler.transform(df).select('features')

# Calculate the correlation matrix
corr_matrix = Correlation.corr(vector_df, 'features').collect()[0][0]

# Output the correlation matrix
print("Correlation matrix:\n", corr_matrix)

# Initialize an empty dictionary to store normalized columns
normalized_columns = {}

# Normalize each column separately and store in the dictionary
for col_name in numeric_cols:
    assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vector")
    assembled_data = assembler.transform(df)
    
    # Initialize MinMaxScaler for each column
    scaler = MinMaxScaler(inputCol=f"{col_name}_vector", outputCol=f"{col_name}_normalized")
    
    # Fit and transform the MinMaxScaler
    normalized_data = scaler.fit(assembled_data).transform(assembled_data)
    
    # Store the normalized column in the dictionary
    normalized_columns[col_name] = normalized_data.select(f"{col_name}_normalized")

# Create a new DataFrame by joining the normalized columns
normalized_df = normalized_columns[numeric_cols[0]]
for col_name in numeric_cols[1:]:
    normalized_df = normalized_df.join(normalized_columns[col_name])

# Show the new DataFrame with normalized columns
normalized_df.show()

# Flatten the DataFrame
flatten_df = normalized_df.select([col(col_name).cast('string') for col_name in normalized_df.columns])

# Write the flattened DataFrame to a CSV file
flatten_df.write.csv('path_to_save_normalized_data', header=True, mode='overwrite')

# Stop the Spark session
spark.stop()




