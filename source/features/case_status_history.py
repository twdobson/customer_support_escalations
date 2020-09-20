# The below code is based on the baseline model's extract features history
# extract_features_history < - function(dt, ref_ids_escalated)

from pyspark.sql import Window
from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.etl.constants_and_parameters import TIME_INTERVAL
from source.utils.ml_tools.categorical_encoders import encode_categorical_using_mean_response_rate_inplace
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

case_status_history = spark_read.parquet(
    path=data_dir.make_interim_path('case_status_history')
)

test = spark_read.parquet(
    path=data_dir.make_interim_path('test')
)

ref_ids_escalated = (
    case_status_history
        .filter(
        F.col("inverse_time_to_next_escalation") > 0
    )
        .select(
        'reference_id'
    )
        .distinct()
)
ref_ids_escalated.count()

history_with_cutoff_times = spark_read.parquet(
    path=data_dir.make_processed_path('history_with_cutoff_times')
)

history_with_cutoff_times.show()

base_table_case_status_history_features = (
    history_with_cutoff_times
        .withColumn(
        'severity_level',
        F.substring(F.col('severity'), 0, 1).cast('int')
    )
        .withColumn(
        'initial_case_severity_level',
        F.first('severity_level').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'cutoff_case_severity_level',
        F.last('severity_level').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'seconds_since_previous_event',
        F.col('seconds_since_case_start') -
        F.lag('seconds_since_case_start', 1).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'cutoff_seconds_since_previous_event',
        F.last('seconds_since_previous_event').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        ) / TIME_INTERVAL
    )
        .withColumn(
        'event_history_frequency',
        F.avg('seconds_since_previous_event').over(
            Window
                .partitionBy('reference_id')
        ) / TIME_INTERVAL
    )
        .withColumn(
        "case_severity_has_changed_between_start_and_end",
        F.when(
            F.col('initial_case_severity_level') != F.col('initial_case_severity_level'), 1
        ).otherwise(
            0
        )
    )

)
base_table_case_status_history_features.show()

# **************************************** COLLATING FEATURES **********************************

response_encoded_metadata_features = (
    base_table_case_status_history_features
        .join(
        ref_ids_escalated
            .withColumn(
            'response',
            F.lit(1)
        )
        ,
        on=['reference_id'],
        how='left'
    )
        .na
        .fill(
        value=0,
        subset='response'
    )
)
columns_to_mean_response_rate_encode = [
    'initial_case_severity_level',
    'cutoff_case_severity_level'
]

mean_encoded_columns = [
    f'mean_response_rate_{col}'
    for col
    in columns_to_mean_response_rate_encode
]

response_encoded_metadata_features = encode_categorical_using_mean_response_rate_inplace(
    df=response_encoded_metadata_features,
    categorical_column="cutoff_case_severity_level",
    response_column='response'
)
response_encoded_metadata_features = encode_categorical_using_mean_response_rate_inplace(
    df=response_encoded_metadata_features,
    categorical_column="initial_case_severity_level",
    response_column='response'
)

### Mean encoding done

first_last_history_features = (

    base_table_case_status_history_features
        .join(
        response_encoded_metadata_features
            .select(
            *['reference_id', 'seconds_since_case_start'] + mean_encoded_columns
        ),
        on=['reference_id', 'seconds_since_case_start'],
        how='inner'
    )
        .select(
        *mean_encoded_columns + [
            'reference_id',
            'initial_case_severity_level',
            'cutoff_case_severity_level',
            'cutoff_seconds_since_previous_event',
            'event_history_frequency',
            'case_severity_has_changed_between_start_and_end'
        ]
    )
        .drop_duplicates()
)
first_last_history_features.show()

# aggregated_history_features = (
#     base_table_case_status_history_features
#         .groupby(
#         'reference_id'
#     )
#         .agg(
#         F.avg('seconds_since_previous_event').alias('average_seconds_since_previous_event')
#     )
# )

base_table_case_status_history_features.filter(F.col('reference_id') == 100052).show(n=100)
history_with_cutoff_times.groupby('severity').count()

first_last_history_features.show()
first_last_history_features.count()

base_table_case_status_history_features.show()

case_status_history_features = (
    first_last_history_features

    #     .join(
    #     aggregated_history_features,
    #     on=['reference_id'],
    #     how='inner'
    # )
)

case_status_history_features.filter(F.col('reference_id') == 100052).show()

spark_write.parquet(
    case_status_history_features,
    path=data_dir.make_feature_path('case_status_history_features'),
    n_partitions=10
)

case_status_history_features = spark_read.parquet(
    path=data_dir.make_feature_path('case_status_history_features')
)

case_status_history_features.show()
print(f'number of rows in case status history features {case_status_history_features.count()}')  # 52968
print(len(case_status_history_features.columns))
