# The below code is based on the baseline model's extract features history
# extract_features_history < - function(dt, ref_ids_escalated)

from pyspark.sql import Window
from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.etl.constants_and_parameters import TIME_INTERVAL
from source.utils.ml_tools.categorical_encoders import (
    one_hot_encode_categorical,
    label_encode_categorical_inplace,
    encode_categorical_using_mean_response_rate_inplace
)
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

comments = spark_read.parquet(
    path=data_dir.make_interim_path('comments')
)
case_status_history = spark_read.parquet(
    path=data_dir.make_interim_path('case_status_history')
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

comments_with_cutoff_times = spark_read.parquet(
    path=data_dir.make_processed_path('comments_with_cutoff_times')
)

comments_with_cutoff_times.show()
comments_with_cutoff_times.groupby('comment_type').count().orderBy(F.desc('count')).show(n=100)

comment_types = [
    'general',
    'programming',
    'email',
    'explanation',
    'reproduction',
    'workaround',
    'configuration',
    'solution',
    'symptom',
    'problem',
    'educreferral'
]

encoded_comments_with_cutoff, one_hot_encoded_comment_columns = (
    one_hot_encode_categorical(
        df=comments_with_cutoff_times,
        categorical_column='comment_type',
        values_to_one_hot_encode=comment_types
    )
)
columns_for_label_encoding = [
    'comment_type',
    'notes'
]

for col in columns_for_label_encoding:
    encoded_comments_with_cutoff = (
        label_encode_categorical_inplace(
            df=encoded_comments_with_cutoff,
            categorical_column=col
        )
    )

for col in one_hot_encoded_comment_columns:
    encoded_comments_with_cutoff = (
        encoded_comments_with_cutoff
            .withColumn(
            f'proportion_comment_type_is_{col}',
            F.avg(col).over(
                Window
                    .partitionBy('reference_id')
            )

        )
    )

# encoded_comments_with_cutoff.show()
# encoded_comments_with_cutoff.groupBy('notes').count().orderBy(F.desc('count')).show()

base_table_case_status_history_features = (
    encoded_comments_with_cutoff
        .withColumn(
        'last_label_encoded_comment_type',
        F.last('label_encoded_comment_type').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'first_label_encoded_comment_type',
        F.first('label_encoded_comment_type').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'last_label_encoded_notes',
        F.last('label_encoded_notes').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'first_label_encoded_notes',
        F.first('label_encoded_notes').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'unique_comments',
        F.size(
            F.collect_set('comment_type').over(
                Window
                    .partitionBy('reference_id')
            )
        )
    )
        .withColumn(
        'unique_global_ids',
        F.size(
            F.collect_set('global_id').over(
                Window
                    .partitionBy('reference_id')
            )
        )
    )
        .withColumn(
        'unique_comments_created_by',
        F.size(
            F.collect_set('created_by').over(
                Window
                    .partitionBy('reference_id')
            )
        )
    )
        .withColumn(
        'last_comment_time',
        F.last('seconds_since_case_start').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'seconds_since_previous_comment',
        F.col('seconds_since_case_start') -
        F.lag('seconds_since_case_start', 1).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'cutoff_seconds_since_previous_comment',
        F.last('seconds_since_previous_comment').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        ) / TIME_INTERVAL
    )
        .withColumn(
        'comment_frequency',
        F.avg('seconds_since_previous_comment').over(
            Window
                .partitionBy('reference_id')
        ) / TIME_INTERVAL
    )
        .withColumn(
        'comments_words_in_notes',
        F.size(
            F.split(
                str=F.col('notes'),
                pattern=' '
            )
        )
    )
        .withColumn(
        'last_comment_note_length',
        F.last('comments_words_in_notes').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'average_comment_words_in_notes',
        F.avg('comments_words_in_notes').over(
            Window
                .partitionBy('reference_id')
        )
    )
)

helper_columns = [
    'comments_words_in_notes',
    'seconds_since_previous_comment'
]

non_distinct_columns = [
    col
    for col
    in base_table_case_status_history_features.columns
    if col not in encoded_comments_with_cutoff.columns + one_hot_encoded_comment_columns + helper_columns
]

base_table_case_status_history_features.count()
base_table_case_status_history_features.select(['reference_id'] + non_distinct_columns).drop_duplicates().show()
base_table_case_status_history_features.select(['reference_id'] + non_distinct_columns).drop_duplicates().count()
comments_with_cutoff_times.select('reference_id').drop_duplicates().count()


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
    'last_label_encoded_comment_type'
    # 'cutoff_case_severity_level'
]

mean_encoded_columns = [
    f'mean_response_rate_{col}'
    for col
    in columns_to_mean_response_rate_encode
]
response_encoded_metadata_features = encode_categorical_using_mean_response_rate_inplace(
    df=response_encoded_metadata_features,
    categorical_column="last_label_encoded_comment_type",
    response_column='response'
)
# response_encoded_metadata_features = encode_categorical_using_mean_response_rate_inplace(
#     df=response_encoded_metadata_features,
#     categorical_column="first_label_encoded_comment_type",
#     response_column='response'
# )

### Mean encoding done

comments_features = (
    base_table_case_status_history_features
        .join(
        response_encoded_metadata_features
            .select(
            *['reference_id', 'seconds_since_case_start'] + mean_encoded_columns
        ),
        on=['reference_id', 'seconds_since_case_start'],
        how='inner'
    )
        .select(*['reference_id'] + non_distinct_columns)
        .drop_duplicates()
)

spark_write.parquet(
    df=comments_features,
    path=data_dir.make_feature_path('comments'),
    n_partitions=10
)

comments_features = spark_read.parquet(
    path=data_dir.make_feature_path('comments')
)
comments_features.show()
print(f'rows in milestone features {comments_features.count()}')  # 52967
print(f'columns in milestone features {len(comments_features.columns)}')

# next_to_last_comment_type = as.character(first(last(COMMENT_TYPE, 2))),
# next_to_last_comment_note_length = length(strsplit(first(last(NOTES, 2)), " ")[[1]]),
# mean_comment_note_length = mean(sapply(strsplit(NOTES, " "), length)),
# unique_terms_in_comment_note = uniqueN(unlist(strsplit(NOTES, " "))),
# unique_terms_in_last_comment_note = uniqueN(strsplit(last(NOTES), " ")[[1]]) ), by = REFERENCEID]
