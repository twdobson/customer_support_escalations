# The below code is based on the baseline model's extract features history
# extract_features_history < - function(dt, ref_ids_escalated)

from pyspark.sql import Window
from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.etl.constants_and_parameters import TIME_INTERVAL
from source.utils.ml_tools.categorical_encoders import label_encode_categorical_inplace
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

test = spark_read.parquet(
    path=data_dir.make_interim_path('test')
)

milestones_with_cutoff_times = spark_read.parquet(
    path=data_dir.make_processed_path('milestones_with_cutoff_times')
)

milestones_with_cutoff_times.show()

milestone_columns_to_drop = [
    'seconds_since_case_start',
    'milestone_id',
    'milestone_description',
    'note_description',
    'created_by',
    'updated_by',
    'inverse_seconds_since_case_start',
    'decision_time'
]

label_encoded_milestones = (
    label_encode_categorical_inplace(
        df=milestones_with_cutoff_times,
        categorical_column='milestone_description'
    )
)

label_encoded_milestones = (
    label_encode_categorical_inplace(
        df=label_encoded_milestones,
        categorical_column='updated_by'
    )
)

base_table_milestone_features = (
    label_encoded_milestones
        .withColumn(
        'reverse_order_row_number_seconds_since_case_start',
        F.row_number().over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.desc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'last_milestone_id',
        F.last('milestone_id').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'second_last_milestone_id',
        F.lag('milestone_id', 2).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'last_milestone_id_time_occurrence',
        F.last('seconds_since_case_start').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'second_last_milestone_id_time_occurrence',
        F.lag('seconds_since_case_start', 2).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'unique_number_of_milestone_ids',
        F.size(
            F.collect_set('milestone_id').over(
                Window
                    .partitionBy('reference_id')
            )
        )
    )
        .withColumn(
        'seconds_since_previous_milestone',
        F.col('seconds_since_case_start') -
        F.lag('seconds_since_case_start', 1).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))

        )
    )
        .withColumn(
        'cutoff_seconds_since_previous_milestone',
        F.last('seconds_since_previous_milestone').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        ) / TIME_INTERVAL
    )
        .withColumn(
        'milestone_frequency',
        F.avg('seconds_since_previous_milestone').over(
            Window
                .partitionBy('reference_id')
        ) / TIME_INTERVAL
    )
        .withColumn(
        'last_milestone_description',
        F.last('milestone_description').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'last_label_encoded_milestone_description',
        F.last('label_encoded_milestone_description').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'average_label_encoded_milestone_description',
        F.avg('label_encoded_milestone_description').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'min_label_encoded_milestone_description',
        F.min('label_encoded_milestone_description').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'max_label_encoded_milestone_description',
        F.max('label_encoded_milestone_description').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'standard_deviation_label_encoded_milestone_description',
        F.stddev('label_encoded_milestone_description').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'length_of_milestone_description',
        F.size(
            F.split(
                str=F.col('last_milestone_description'),
                pattern=' '
            )
        )
    )
        .withColumn(
        'average_length_of_milestone_descriptions',
        F.avg("length_of_milestone_description").over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'min_length_of_milestone_descriptions',
        F.min("length_of_milestone_description").over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'max_length_of_milestone_descriptions',
        F.max("length_of_milestone_description").over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'length_of_last_milestone_description',
        F.last('length_of_milestone_description').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))

        )
    )
        .withColumn(
        'length_of_second_last_milestone_description',
        F.lag('length_of_milestone_description', 2).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))

        )
    )

        .withColumn(
        'is_milestone_note_6763',
        F.when(
            F.col('note_description') == 6763, 1
        ).otherwise(
            0
        )
    )
        .withColumn(
        'note_description_null_for_6763',
        F.when(
            F.col('note_description') == 6763, None
        ).otherwise(
            F.col('note_description')
        )
    )
        .withColumn(
        'proportion_is_milestone_note_6763',
        F.avg('note_description_null_for_6763').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'length_of_milestone_note',
        F.size(
            F.split(
                str=F.col('note_description'),
                pattern=' '
            )
        )
    )
        .withColumn(
        'length_of_milestone_note_null_for_6763',
        F.size(
            F.split(
                str=F.col('note_description_null_for_6763'),
                pattern=' '
            )
        )
    )
        .withColumn(
        'average_length_of_milestone_note',
        F.avg('length_of_milestone_note').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'average_length_of_note_description_null_for_6763',
        F.avg('note_description_null_for_6763').over(
            Window
                .partitionBy('reference_id')
        )
    )
        .withColumn(
        'last_length_of_milestone_note',
        F.last('length_of_milestone_note').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )
        .withColumn(
        'second_last_length_of_milestone_note',
        F.lag('length_of_milestone_note', 2).over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
        )
    )
        .withColumn(
        'last_length_of_note_description_null_for_6763',
        F.last('length_of_milestone_note_null_for_6763').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )

    )
        .na
        .fill(
        value=-999,
        subset=['last_length_of_note_description_null_for_6763']
    )
        .withColumn(
        'last_label_encoded_updated_by',
        F.last('label_encoded_updated_by').over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.asc('seconds_since_case_start'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    )

)

milestone_features = (
    base_table_milestone_features
    .filter(
        F.col('reverse_order_row_number_seconds_since_case_start')==1
    )
        .drop(
        *milestone_columns_to_drop
    ).drop(
        *[
            'seconds_since_previous_milestone',
            'length_of_milestone_description',
            'is_milestone_note_6763',
            'note_description_null_for_6763',
            'length_of_milestone_note',
            'length_of_milestone_note_null_for_6763',
            'label_encoded_updated_by',
            'label_encoded_milestone_description',
            'last_milestone_description'
        ]
    )
        .drop_duplicates()
)

spark_write.parquet(
    df=milestone_features,
    path=data_dir.make_feature_path('milestone'),
    n_partitions=10
)

milestone_features = spark_read.parquet(
    path=data_dir.make_feature_path('milestone')
)
milestone_features.show()
print(f'rows in milestone features {milestone_features.count()}')

# unique_terms_in_milestone_desc = uniqueN(unlist(strsplit(MILESTONEDESCRIPTION, " "))),
# mean_unique_terms_in_milestone_desc = mean(sapply(strsplit(MILESTONEDESCRIPTION, " "), uniqueN)),
# unique_terms_in_last_milestone_desc = uniqueN(strsplit(last(MILESTONEDESCRIPTION), " ")[[1]]),
#
# # next_to_last_milestone_note_length = length(strsplit(first(last(NOTEDESCRIPTION, 2)), " ")[[1]]),
# unique_terms_in_milestone_notes = uniqueN(unlist(strsplit(NOTEDESCRIPTION, " "))),
# mean_unique_terms_in_milestone_notes = mean(sapply(strsplit(NOTEDESCRIPTION, " "), uniqueN)),
# unique_terms_in_last_milestone_note = uniqueN(strsplit(last(NOTEDESCRIPTION), " ")[[1]]),
# last_milestone_updated_by = last(UPDATED_BY) ), by = REFERENCEID]
