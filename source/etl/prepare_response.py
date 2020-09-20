from pyspark.sql import Window
from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
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
).withColumn(
    'inverse_time_to_next_escalation',
    F.col('inverse_time_to_next_escalation').cast('double')
)

milestones = spark_read.parquet(
    path=data_dir.make_interim_path('milestones')
)

comments = spark_read.parquet(
    path=data_dir.make_interim_path('comments')
)

case_status_history.show()

escalation_starts = (
    case_status_history
        .filter(
        F.col('is_escalate') == 'Y'
    )
        .groupby(
        'reference_id'
    )
        .agg(
        F.min('seconds_since_case_start').alias('escalation_start'),
        F.max('seconds_since_case_start').alias('case_end')
    )
)

escalation_starts.count()  # 646
escalation_starts.filter(F.col('reference_id') == 100087).show()

escalation_points_distribution = (
    escalation_starts
        .withColumn(
        'escalation_point_relative_to_case_duration',
        F.col('escalation_start') / F.col('case_end')
    )
        .withColumn(
        'percent_rank_escalation_point_relative_to_case_duration',
        F.percent_rank().over(
            Window
                .partitionBy()
                .orderBy(F.asc("escalation_point_relative_to_case_duration"))
        )
    )
        .withColumn(
        'percentile',
        F.floor(F.col('percent_rank_escalation_point_relative_to_case_duration') * 100)
    )
        .groupBy(
        'percentile'
    )
        .agg(
        F.avg('escalation_point_relative_to_case_duration').alias('average_percentile_escalation_point')
    )
)
escalation_points_distribution.withColumn('r', F.round(F.col("average_percentile_escalation_point"), 2)).show(n=100)

escalation_training_ids = (
    escalation_starts
        .select(
        'reference_id'
    )
        .distinct()
)
escalation_training_ids.count()

escalation_training_targets = (
    case_status_history
        .join(
        escalation_starts,
        on=(case_status_history['reference_id'] == escalation_starts['reference_id']) &
           (case_status_history['seconds_since_case_start'] < escalation_starts['escalation_start']),
        how='inner'
    )
        .groupby(
        case_status_history['reference_id']
    )
        .agg(
        F.max('seconds_since_case_start').alias('decision_time'),
        F.max('inverse_time_to_next_escalation').alias('response')
    )
)

escalation_training_targets.filter(F.col('reference_id') == 100227).show()  # |      100227|   0|0.991052993805919|
escalation_training_targets.filter(F.col('reference_id') == 100239).show()  # |      100239| 7864270|0.975202320620337|

escalation_training_targets.count()  # 646

######################### NON ESCALATION ##################################################################
###########################################################################################################

non_escalation_case_status_history = (
    case_status_history
        .join(
        escalation_starts,
        on=['reference_id'],
        how='left_anti'
    )
)
non_escalation_case_status_history.count()  # 783586

non_escalation_decision_times = (
    non_escalation_case_status_history
        .groupby(
        'reference_id'
    )
        .agg(
        F.max('seconds_since_case_start').alias('case_end')
    )
        .crossJoin(
        escalation_points_distribution
    )
        .withColumn(
        'time_cut',
        F.col('case_end') * F.col("average_percentile_escalation_point")
    )
        .withColumn(
        'random_row_rank_for_sampling',
        F.row_number().over(
            Window
                .partitionBy('reference_id')
                .orderBy(F.rand())
        )
    )
        .filter(
        F.col('random_row_rank_for_sampling') == 1
    )
)
non_escalation_case_status_history.count()  # 783586
non_escalation_decision_times.count()  # 52989

non_escalation_training_targets = (
    non_escalation_decision_times
        .join(
        non_escalation_case_status_history,
        on=['reference_id'],
        how='inner'
    )
        .filter(
        F.col('seconds_since_case_start') < F.col('time_cut')
    )
        .groupBy(
        'reference_id'
    )
        .agg(
        F.max('seconds_since_case_start').alias('decision_time')
    )
        .withColumn(
        'target',
        F.lit(0)
    )
)

non_escalation_training_targets.show()
non_escalation_training_targets.count()  # 51443 (we loose 52989 - 51443 = 1546)
non_escalation_training_targets.groupBy().agg(F.avg('decision_time').alias('average_decision_time')).show()  # 1164225

# TODO see who we are dropping here and why
(
    non_escalation_decision_times
        .join(
        non_escalation_case_status_history,
        on=['reference_id'],
        how='inner'
    )
        .filter(
        F.col('seconds_since_case_start') == F.col('time_cut')
    )
        .count()  # 1545
)

train_validation_targets = (
    escalation_training_targets
        .union(
        non_escalation_training_targets
    )
        .join(
        test
            .select('reference_id'),
        on=['reference_id'],
        how='left_anti'
    )
)
train_validation_targets.show()
train_validation_targets.count()

train_validation_targets.groupBy().agg(F.avg('response')).show()

response_file = (
    test
        .select(
        'reference_id',
        F.col('seconds_since_case_start').alias('decision_time'),
        F.col('inverse_time_to_next_escalation').alias('response')
    )
        .union(
        train_validation_targets
    )
)

response_file.show()
response_file.count()  # 52968
test.groupBy().agg(F.avg('seconds_since_case_start').alias('average_decision_time')).show()  # 729856.2922037095

spark_write.parquet(
    response_file,
    path=data_dir.make_processed_path('model_files', 'response_file'),
    n_partitions=10
)

decision_times = (
    response_file
        .drop(
        'response'
    )
)

case_status_history.count()
case_status_history.drop_duplicates().count()

history_with_cutoff_times = (
    case_status_history
        .join(
        decision_times,
        on=['reference_id'],
        how='inner'
    )
        .filter(
        F.col('seconds_since_case_start') <= F.col('decision_time')
    )
)

spark_write.parquet(
    df=history_with_cutoff_times,
    path=data_dir.make_processed_path('history_with_cutoff_times'),
    n_partitions=10
)

history_with_cutoff_times.count()

milestones_with_cutoff_times = (
    milestones
        .join(
        decision_times,
        on=['reference_id'],
        how='inner'
    )
        .filter(
        F.col('seconds_since_case_start') <= F.col('decision_time')
    )
)

milestones_with_cutoff_times.show()
milestones_with_cutoff_times.count()

spark_write.parquet(
    df=milestones_with_cutoff_times,
    path=data_dir.make_processed_path('milestones_with_cutoff_times'),
    n_partitions=10
)

milestones_with_cutoff_times = spark_read.parquet(
    data_dir.make_processed_path('milestones_with_cutoff_times')
)

(
    milestones_with_cutoff_times
        .filter(
        F.col('reference_id') == 100052
    )
        .show()
)
(
    milestones_with_cutoff_times
        .filter(
        F.col('reference_id') == 100052
    )
        .count()
)

# 100052

comments_with_cutoff_times = (
    comments
        .join(
        decision_times,
        on=['reference_id'],
        how='inner'
    )
        .filter(
        F.col('seconds_since_case_start') <= F.col('decision_time')
    )
)

comments_with_cutoff_times.show()
comments_with_cutoff_times.count()

spark_write.parquet(
    df=comments_with_cutoff_times,
    path=data_dir.make_processed_path('comments_with_cutoff_times'),
    n_partitions=10
)

comments_with_cutoff_times = spark_read.parquet(
    data_dir.make_processed_path('comments_with_cutoff_times')
)

comments_with_cutoff_times.show()
print(f"rows in table comments_with_cutoff_times {comments_with_cutoff_times.count()}")
