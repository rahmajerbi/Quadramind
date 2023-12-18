from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, avg
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
from pyspark.sql.functions import lag, lead, when, lit
from pyspark.sql.functions import coalesce

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

zero_sbp_dbp_rows = df.filter((df['SBP'] == 0) & (df['DBP'] == 0) & (df['HR'] == 0) & (df['SPO2'] == 0))
print('ZERO BLOOD PRESSURE: \n')
zero_sbp_dbp_rows.show()


def replace_zero_SBP_DBP(df):
    # Calculate lag and lead values for SBP and DBP columns
    df = df.withColumn('SBP_lag', lag('SBP').over(Window.orderBy('DateTime')))
    df = df.withColumn('SBP_lead', lead('SBP').over(Window.orderBy('DateTime')))
    df = df.withColumn('DBP_lag', lag('DBP').over(Window.orderBy('DateTime')))
    df = df.withColumn('DBP_lead', lead('DBP').over(Window.orderBy('DateTime')))

    # Replace NULL in SBP_lag and DBP_lag with zeros
    df = df.withColumn('SBP_lag', coalesce(df['SBP_lag'], lit(0)))
    df = df.withColumn('DBP_lag', coalesce(df['DBP_lag'], lit(0)))

    # Replace zero SBP and DBP values with mean of t-1 and t+1 rows if HR and SPO2 are not zero
    df = df.withColumn('SBP', when((df['SBP'] == 0) & (df['HR'] != 0) & (df['SPO2'] != 0),
                                   (df['SBP_lag'] + df['SBP_lead']) / 2).otherwise(df['SBP']))
    df = df.withColumn('DBP', when((df['DBP'] == 0) & (df['HR'] != 0) & (df['SPO2'] != 0),
                                   (df['DBP_lag'] + df['DBP_lead']) / 2).otherwise(df['DBP']))

    # Drop intermediate columns
    df = df.drop('SBP_lag', 'SBP_lead', 'DBP_lag', 'DBP_lead')

    return df



# Apply the function to your DataFrame
modified_df = replace_zero_SBP_DBP(df)
modified_df.show()

# Filter rows where SBP and DBP have zero values
zero_sbp_dbp_rows = modified_df.filter((modified_df['SBP'] == 0) & (modified_df['DBP'] == 0))
print('ZERO BLOOD PRESSURE: \n')
zero_sbp_dbp_rows.show()


def BloodPressureClassification(df):
    df = df.withColumn(
            'BP_level',
            when((col('SBP') < 120) & (col('DBP') < 80), lit("Normal"))
            .when((col('SBP') >= 120) & (col('SBP') < 130) & (col('DBP') < 80), lit("Elevated"))
            .when((col('SBP') >= 130) & (col('SBP') < 140) | ((col('DBP') >= 80) & (col('DBP') < 90)), lit("Stage 1 Hypertension"))
            .when((col('SBP') >= 140) | (col('DBP') >= 90), lit("Stage 2 Hypertension"))
            .otherwise(lit("Hypertensive Crisis"))
        )
    return df


modified_df = BloodPressureClassification(modified_df)
modified_df.show()
modified_df.coalesce(1).write.option("header", "true").csv("./Kafka/data/proccesed_mimic_2425.csv")


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

# Assuming the 'DateTime' column is already in the timestamp format, if not, convert it
data = df.withColumn('DateTime', df['DateTime'].cast('timestamp'))

# Define the window partitioned by time
time_window = Window.orderBy(col('DateTime').cast('long')).rangeBetween(-300, 0)  # 300 seconds (5 min) window

# Calculate rolling averages for columns RR, SPO2, MAP, SBP, DBP, HR, PP, CO over the defined window
rolling_avg_df = data.withColumn('RR_RollingAvg', avg(col('RR')).over(time_window)) \
    .withColumn('SPO2_RollingAvg', avg(col('SPO2')).over(time_window)) \
    .withColumn('MAP_RollingAvg', avg(col('MAP')).over(time_window)) \
    .withColumn('SBP_RollingAvg', avg(col('SBP')).over(time_window)) \
    .withColumn('DBP_RollingAvg', avg(col('DBP')).over(time_window)) \
    .withColumn('HR_RollingAvg', avg(col('HR')).over(time_window)) \
    .withColumn('PP_RollingAvg', avg(col('PP')).over(time_window)) \
    .withColumn('CO_RollingAvg', avg(col('CO')).over(time_window))

# Show the resulting DataFrame with rolling averages
rolling_avg_df.show()

# Convert Spark DataFrame to Pandas for plotting
rolling_avg_pandas = rolling_avg_df.toPandas()

# Plotting
plt.figure(figsize=(12, 8))

# Plot original columns
plt.plot(rolling_avg_pandas['DateTime'], rolling_avg_pandas['SBP'], label='SBP')
# Plot rolling averages
plt.plot(rolling_avg_pandas['DateTime'], rolling_avg_pandas['SBP_RollingAvg'], label='SBP Rolling Avg')
# The rolling average calculations are computed over a 1-hour window for each timestamp.
# Plotting
plt.figure(figsize=(12, 8))

# Plot original columns
plt.plot(rolling_avg_pandas['DateTime'], rolling_avg_pandas['DBP'], label='DBP')
# Plot rolling averages
plt.plot(rolling_avg_pandas['DateTime'], rolling_avg_pandas['DBP_RollingAvg'], label='DBP Rolling Avg')

# Customize plot
plt.xlabel('DateTime')
plt.ylabel('Values')
plt.title('Time Series Data and Rolling Averages')
plt.legend()
plt.show()

# Stop the Spark session
spark.stop()


## OUTPUT ##

# +-------------------+----+-----+----+----+----+-----+----+-------+
# |           DateTime|  RR| SPO2| MAP| SBP| DBP|   HR|  PP|     CO|
# +-------------------+----+-----+----+----+----+-----+----+-------+
# |2020-10-18 15:24:25|35.0| 99.9| 0.0| 0.0| 0.0|106.9| 0.0|    0.0|
# |2020-10-18 15:25:25|36.4|100.0|87.0|98.9|63.1|107.3|35.8|3841.34|
# |2020-10-18 15:26:25|35.2|100.0|75.2|97.9|63.0|107.5|34.9|3751.75|
# |2020-10-18 15:27:25|34.0|100.0|74.8|97.2|62.5|107.0|34.7| 3712.9|
# |2020-10-18 15:28:25|34.9|100.0|74.0|96.0|62.0|107.0|34.0| 3638.0|
# +-------------------+----+-----+----+----+----+-----+----+-------+
# only showing top 5 rows


# Number of rows in the DataFrame: 12877

# +-------+------------------+------------------+-----------------+------------------+------------------+------------------+------------------+------------------+
# |summary|                RR|              SPO2|              MAP|               SBP|               DBP|                HR|                PP|                CO|
# +-------+------------------+------------------+-----------------+------------------+------------------+------------------+------------------+------------------+
# |  count|             12877|             12877|            12877|             12877|             12877|             12877|             12877|             12877|
# |   mean|16.900497010173222| 96.95497398462365|70.49208666614892| 96.91044497942087| 54.56479770132787| 87.33266288731858| 42.34564727809269|3675.1487077735433|
# | stddev| 6.689417514296674|7.4612235341124125| 9.64258275931374|19.424266322745364|10.350134176778717|12.355479187443489|11.114098473613113|  939.202069525275|
# |    min|               0.0|               0.0|              0.0|               0.0|               0.0|               0.0|               0.0|               0.0|
# |    max|              51.8|             100.0|            270.0|             224.8|             118.9|             121.0|             105.9|           8768.52|
# +-------+------------------+------------------+-----------------+------------------+------------------+------------------+------------------+------------------+

# Column 'DateTime' has data type 'timestamp'
# Column 'RR' has data type 'double'
# Column 'SPO2' has data type 'double'
# Column 'MAP' has data type 'double'
# Column 'SBP' has data type 'double'
# Column 'DBP' has data type 'double'
# Column 'HR' has data type 'double'
# Column 'PP' has data type 'double'
# Column 'CO' has data type 'double'
# Missing values for each column: 

# +---+----+---+---+---+---+---+---+
# | RR|SPO2|MAP|SBP|DBP| HR| PP| CO|
# +---+----+---+---+---+---+---+---+
# |  0|   0|  0|  0|  0|  0|  0|  0|
# +---+----+---+---+---+---+---+---+

# Correlation matrix:

#  DenseMatrix([[ 1.        ,  0.11713022,  0.27863902,  0.2247959 ,  0.13482729,
#                0.13756076,  0.26731946,  0.31107062],
#              [ 0.11713022,  1.        ,  0.40331335,  0.28281126,  0.29323193,
#                0.32456546,  0.22119756,  0.1979043 ],
#              [ 0.27863902,  0.40331335,  1.        ,  0.20246111,  0.17519317,
#                0.2092313 ,  0.19069344,  0.18234492],
#              [ 0.2247959 ,  0.28281126,  0.20246111,  1.        ,  0.89757625,
#                0.00748702,  0.91183569,  0.87248197],
#              [ 0.13482729,  0.29323193,  0.17519317,  0.89757625,  1.        ,
#                0.19399362,  0.63744496,  0.67736593],
#              [ 0.13756076,  0.32456546,  0.2092313 ,  0.00748702,  0.19399362,
#                1.        , -0.16757366,  0.22763685],
#              [ 0.26731946,  0.22119756,  0.19069344,  0.91183569,  0.63744496,
#               -0.16757366,  1.        ,  0.89404406],
#              [ 0.31107062,  0.1979043 ,  0.18234492,  0.87248197,  0.67736593,
#                0.22763685,  0.89404406,  1.        ]])


# +--------------------+--------------------+--------------+--------------+--------------+--------------------+-------------+--------------------+
# |       RR_normalized|     SPO2_normalized|MAP_normalized|SBP_normalized|DBP_normalized|       HR_normalized|PP_normalized|       CO_normalized|
# +--------------------+--------------------+--------------+--------------+--------------+--------------------+-------------+--------------------+
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|               [0.0]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]| [0.438083051643835]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.42786582000155...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.4234351977300616]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.4148932773147577]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.4116817889449987]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.40642890704474...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3968833965138928]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3935213696268013]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3719213732762199]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.39035777987619...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3897282551673486]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3981333223850775]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3945295215156035]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.40924580202816...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.42711540830151...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.42149758454106...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.42150784853088...|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.4063274075898783]|
# |[0.6756756756756757]|[0.9990000000000001]|         [0.0]|         [0.0]|         [0.0]|[0.8834710743801654]|        [0.0]|[0.3917764913577206]|
# +--------------------+--------------------+--------------+--------------+--------------+--------------------+-------------+--------------------+
# only showing top 20 rows

# +-------------------+----+-----+----+----+----+-----+----+-------+------------------+-----------------+-----------------+-----------------+------------------+------------------+------------------+------------------+
# |           DateTime|  RR| SPO2| MAP| SBP| DBP|   HR|  PP|     CO|     RR_RollingAvg|  SPO2_RollingAvg|   MAP_RollingAvg|   SBP_RollingAvg|    DBP_RollingAvg|     HR_RollingAvg|     PP_RollingAvg|     CO_RollingAvg|
# +-------------------+----+-----+----+----+----+-----+----+-------+------------------+-----------------+-----------------+-----------------+------------------+------------------+------------------+------------------+
# |2020-10-18 15:24:25|35.0| 99.9| 0.0| 0.0| 0.0|106.9| 0.0|    0.0|              35.0|             99.9|              0.0|              0.0|               0.0|             106.9|               0.0|               0.0|
# |2020-10-18 15:25:25|36.4|100.0|87.0|98.9|63.1|107.3|35.8|3841.34|              35.7|            99.95|             43.5|            49.45|             31.55|             107.1|              17.9|           1920.67|
# |2020-10-18 15:26:25|35.2|100.0|75.2|97.9|63.0|107.5|34.9|3751.75| 35.53333333333334|99.96666666666665|54.06666666666666|65.60000000000001| 42.03333333333333|107.23333333333333|23.566666666666663|           2531.03|
# |2020-10-18 15:27:25|34.0|100.0|74.8|97.2|62.5|107.0|34.7| 3712.9|35.150000000000006|           99.975|            59.25|             73.5|             47.15|           107.175|26.349999999999998|         2826.4975|
# |2020-10-18 15:28:25|34.9|100.0|74.0|96.0|62.0|107.0|34.0| 3638.0| 35.10000000000001|99.97999999999999|             62.2|             78.0|             50.12|107.14000000000001|27.879999999999995|          2988.798|
# |2020-10-18 15:29:25|32.9|100.0|73.5|95.5|61.7|106.8|33.8|3609.84| 34.73333333333334|99.98333333333333|64.08333333333333|80.91666666666667|52.050000000000004|107.08333333333333|28.866666666666664|3092.3050000000003|
# |2020-10-18 15:30:25|35.3|100.0|73.1|94.8|61.4|106.7|33.4|3563.78|34.814285714285724|99.98571428571428|65.37142857142858|82.89999999999999|53.385714285714286|107.02857142857144|29.514285714285712|3159.6585714285716|
# |2020-10-18 15:31:25|31.7|100.0|71.9|93.3|60.5|106.1|32.8|3480.08|34.425000000000004|          99.9875|          66.1875|84.19999999999999|            54.275|106.91250000000001|29.924999999999997|3199.7112500000003|
# |2020-10-18 15:32:25|34.5|100.0|71.6|92.8|60.4|106.5|32.4| 3450.6| 34.43333333333334|99.98888888888888|66.78888888888889|85.15555555555554| 54.95555555555555|106.86666666666667|30.199999999999996| 3227.587777777778|
# |2020-10-18 15:33:25|26.2|100.0|68.5|88.9|57.9|105.2|31.0| 3261.2|             33.61|            99.99|66.96000000000001|85.52999999999999|             55.25|             106.7|30.279999999999994|          3230.949|
# |2020-10-18 15:34:25|33.5|100.0|72.0|93.2|61.0|106.3|32.2|3422.86|              33.6| 99.9909090909091|67.41818181818182|86.22727272727272| 55.77272727272727|106.66363636363636| 30.45454545454545|3248.3954545454544|
# |2020-10-18 15:35:25|25.8|100.0|71.8|93.0|60.7|105.8|32.3|3417.34|             32.95|99.99166666666667|67.78333333333333|86.79166666666667| 56.18333333333334|106.59166666666665| 30.60833333333333| 3262.474166666667|
# |2020-10-18 15:36:25|29.4| 99.8|71.4|93.1|59.5|103.9|33.6|3491.04|32.676923076923075|99.97692307692309|68.06153846153846|87.27692307692307| 56.43846153846154|106.38461538461539|30.838461538461537|3280.0561538461543|
# |2020-10-18 15:37:25|19.5|100.0|70.7|92.8|59.6|104.2|33.2|3459.44|31.735714285714288|99.97857142857143|            68.25|87.67142857142856| 56.66428571428572|106.22857142857143|31.007142857142856| 3292.869285714286|
# |2020-10-18 15:38:25|20.4|100.0|72.3|94.6|61.0|106.8|33.6|3588.48|             30.98|            99.98|            68.52|88.13333333333331| 56.95333333333334|106.26666666666667|             31.18|3312.5766666666673|
# |2020-10-18 15:39:25|28.9|100.0|73.9|97.1|62.0|106.7|35.1|3745.17|30.849999999999998|         99.98125|         68.85625|88.69374999999998|57.268750000000004|         106.29375|            31.425|3339.6137500000004|
# |2020-10-18 15:40:25|21.5|100.0|73.3|96.2|61.3|105.9|34.9|3695.91|30.299999999999994|99.98235294117647|69.11764705882354|89.13529411764705| 57.50588235294118|106.27058823529413|31.629411764705885| 3360.572352941177|
# |2020-10-18 15:41:25|25.5|100.0|72.9|96.0|61.0|105.6|35.0| 3696.0|30.033333333333328|99.98333333333333|69.32777777777778|89.51666666666665|57.699999999999996|106.23333333333333| 31.81666666666667|3379.2072222222228|
# |2020-10-18 15:42:25|21.2|100.0|71.8|94.1|60.2|105.1|33.9|3562.89|29.568421052631578|99.98421052631579| 69.4578947368421|89.75789473684209| 57.83157894736842|106.17368421052632|31.926315789473687| 3388.874736842106|
# |2020-10-18 15:43:25|16.0|100.0|69.5|91.2|58.2|104.1|33.0| 3435.3|28.889999999999997|           99.985|69.46000000000001|89.82999999999998|             57.85|106.07000000000001|             31.98| 3391.196000000001|
# +-------------------+----+-----+----+----+----+-----+----+-------+------------------+-----------------+-----------------+-----------------+------------------+------------------+------------------+------------------+
# only showing top 20 rows
