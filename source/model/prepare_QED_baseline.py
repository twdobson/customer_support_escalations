# The below code is based on the baseline model's extract features history
# extract_features_history < - function(dt, ref_ids_escalated)

from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

milestone_features = spark_read.parquet(
    path=data_dir.make_feature_path('milestone')
)
print(f'rows in milestone features {milestone_features.count()}')
comments_features = spark_read.parquet(
    path=data_dir.make_feature_path('comments')
)
print(f'rows in milestone features {comments_features.count()}')

case_status_history_features = spark_read.parquet(
    path=data_dir.make_feature_path('case_status_history_features')
)
print(f'rows in case_status_history features {case_status_history_features.count()}')

metadata_features = spark_read.parquet(
    path=data_dir.make_feature_path('metadata')
)

response_file = spark_read.parquet(
    path=data_dir.make_processed_path('model_files', 'response_file')
)


print(f'rows in case_status_history features {response_file.count()}')


k_folds = 5

validation_folds = (
    response_file
        .select(
        'reference_id'
    )
        .distinct()
        .withColumn(
        'cross_validation_fold_0',
        F.floor(F.rand(seed=39) * k_folds)
    )
        .withColumn(
        'cross_validation_fold_1',
        F.floor(F.rand(seed=12) * k_folds)
    )
)


model_file = (
    response_file
        .join(
        milestone_features,
        on=['reference_id'],
        how='left'
    )
        .join(
        comments_features,
        on=['reference_id'],
        how='left'
    )
        .join(
        case_status_history_features,
        on=['reference_id'],
        how='left'
    )
    .join(
        metadata_features,
        on=['reference_id'],
        how='inner'
    )
    .withColumn(
        'in_train',
        F.when(
            F.col('response').isNull(), 0
        ).otherwise(
            1
        )
    )
        .join(
        validation_folds,
        on=['reference_id'],
        how='inner'
    )
)

print(f"rows in model file {model_file.count()}")

train_validate_file = (
    model_file
        .filter(
        F.col('in_train') == 1
    )
)

test_file = (
    model_file
        .filter(
        F.col('in_train') == 0
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
print(f"columns in model file {len(test_file.columns)}")
