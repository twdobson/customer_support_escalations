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
escalation_points_distribution.withColumn('r', F.round(F.col("average_percentile_escalation_point"), 1)).show(n=30)

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

escalation_training_targets.filter(F.col('reference_id') == 100227).show()

non_escalation_case_status_history = (
    case_status_history
        .join(
        escalation_starts,
        on=['reference_id'],
        how='left_anti'
    )
)
non_escalation_case_status_history.count()

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
non_escalation_case_status_history.count()
non_escalation_decision_times.count()

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
non_escalation_training_targets.count()

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

history_with_cutoff_times.count() 


# other_training_targets[, target := 0]
#
# training_targets <- rbind(training_targets, other_training_targets)
# training_targets <- training_targets[!(REFERENCEID %in% test_ids)]
# setkey(training_targets, REFERENCEID)
#
# dim(training_targets)
# head(training_targets)
#
# training_targets[, mean(target)]
