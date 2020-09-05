# Baseline model
# 1. Given all products that person has (product_1 - product_21), excluding the prediction class
# 2. Given person demographics - age, year of birth join year
# 3. make prediction

# Product features
# Demographic based product features

# from config.spark_setup import launch_spark

from pyspark.sql import functions as F
from config.data_paths import data_dir
from config.env import *
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

response_file = spark_read.parquet(
    data_dir.make_interim_path('response_file')
)

response_file.orderBy('reference_id', 'seconds_since_case_start').show()

pivoted_milestone_ids_features = spark_read.parquet(
    path=data_dir.make_feature_path('pivoted_milestone_ids'),
)

pivoted_milestone_ids_features.show()

print(f"rows in response file {response_file.count()}")  # 720938
print(f"rows in pivoted milestone features {pivoted_milestone_ids_features.count()}")

model_file = (
    response_file
        .join(
        pivoted_milestone_ids_features,
        on=['reference_id', 'seconds_since_case_start'],
        how='inner'
    )
)

print(f"rows in model file {model_file.count()}")


train_validate_file = (
    model_file
    .filter(
        F.col('in_train')==1
    )
)

test_file = (
    model_file
    .filter(
        F.col('in_train')==0
    )
)

spark_write.parquet(
    df=train_validate_file,
    path=data_dir.make_processed_path('model_files', 'train_validate_file'),
    n_partitions=10
)

spark_write.parquet(
    df=test_file,
    path=data_dir.make_processed_path('model_files', 'test_file'),
    n_partitions=10
)

train_validate_file = spark_read.parquet(
    data_dir.make_processed_path('model_files', 'train_validate_file')
)

test_file = spark_read.parquet(
    data_dir.make_processed_path('model_files', 'test_file')
)

print(f"rows in train validate file {train_validate_file.count()}")
print(f"rows in test file {test_file.count()}")