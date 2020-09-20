# Baseline model
# 1. Given all products that person has (product_1 - product_21), excluding the prediction class
# 2. Given person demographics - age, year of birth join year
# 3. make prediction

# Product features
# Demographic based product features

# from config.spark_setup import launch_spark

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

response_file = spark_read.parquet(
    data_dir.make_interim_path('response_file')
)
milestones = spark_read.parquet(
    data_dir.make_interim_path('milestones')
)

milestones.orderBy('reference_id', 'seconds_since_case_start').show()

# EDA milestons
# milestone_id distributions
milestones.create_distribution(
    groupby_columns=['milestone_id'],
    numeric_column='seconds_since_case_start',
    include_count_distribution=True
).orderBy(F.desc('count')).show(n=70)

# null summary
milestones.create_null_summary().show()

# unique summary
milestones.create_unique_value_summary(columns=milestones.columns).show(truncate=False)

milestones_with_prior_mile_stones = (
    milestones
        .select(
        F.col('reference_id'),
        F.col('seconds_since_case_start'),
        F.col('milestone_id'),
    )
        .join(
        milestones
            .withColumnRenamed(
            'reference_id',
            'prior_reference_id'
        )
            .withColumnRenamed(
            'seconds_since_case_start',
            'prior_seconds_since_case_start'
        )
            .withColumnRenamed(
            'milestone_id',
            'prior_milestone_id'
        )
        ,
        on=(F.col('reference_id') == F.col('prior_reference_id')) &
           (F.col('seconds_since_case_start') > F.col("prior_seconds_since_case_start"))
    )
)

milestones_with_prior_mile_stones.orderBy('reference_id', 'seconds_since_case_start').show()

# ************************* FEATURES PIVOTED MILESTONE IDS ******************************

pivoted_milestone_ids = (
    milestones_with_prior_mile_stones
        .groupby(
        'reference_id',
        F.col('seconds_since_case_start').alias('milestone_seconds_since_case_start')
    )
        .pivot(
        "prior_milestone_id"
    )
        .agg(
        F.count('*').alias('count_milestone'),
        F.min('seconds_since_case_start').alias('first_time_since_case_start'),
        F.max('seconds_since_case_start').alias('last_time_since_start')

    )
)

total_milestones = (
    milestones_with_prior_mile_stones
        .groupby(
        'reference_id',
        'seconds_since_case_start'
    )
        .agg(
        F.count('*').alias('total_milestones')
    )
)

test = (
    response_file
        .select(
        'reference_id',
        'seconds_since_case_start',
        'in_train'
    )
        .join(
        pivoted_milestone_ids,
        on=['reference_id'],
        how='left'
    )
        .join(
        total_milestones,
        on=['reference_id', 'seconds_since_case_start'],
        how='left'
    )
)

filtered_test = (
    test
        .filter(
        (F.col('milestone_seconds_since_case_start') <= F.col('seconds_since_case_start')) |
        (F.col('seconds_since_case_start') == 0) |
        (F.col('total_milestones').isNull())
    )
        .withColumn(
        'seconds_to_closest_milestone_features',
        F.col('seconds_since_case_start') - F.col('milestone_seconds_since_case_start')
    )
        .withColumn(
        'rank_seconds_to_milestone_features',
        F.row_number().over(
            Window
                .partitionBy('reference_id', 'seconds_since_case_start')
                .orderBy("seconds_to_closest_milestone_features")
        )
    )
        .filter(
        F.col('rank_seconds_to_milestone_features') == 1
    )
)
#
test.select('reference_id', 'seconds_since_case_start').filter(F.col('in_train') == 1).distinct().count()
# test.select('reference_id', 'seconds_since_case_start').count()
filtered_test.select('reference_id', 'seconds_since_case_start').filter(F.col('in_train') == 1).distinct().count()
# filtered_test.select('reference_id', 'seconds_since_case_start').distinct().count()
#
test.select('reference_id', 'seconds_since_case_start').filter(F.col('in_train') == 0).distinct().count()
filtered_test.select('reference_id', 'seconds_since_case_start').filter(F.col('in_train') == 0).distinct().count()

pivoted_milestone_ids_features = (
    response_file
        .select(
        'reference_id',
        'seconds_since_case_start'
    )
        .join(
        pivoted_milestone_ids
            .fillna(
            value=0,
            subset=[col for col in pivoted_milestone_ids.columns if 'count' in col]
        )
            .fillna(
            value=-999,
            subset=[col for col in pivoted_milestone_ids.columns if 'count' in col]
        )

        ,
        on=['reference_id'],
        how='left'
    )
        .join(
        total_milestones,
        on=['reference_id', 'seconds_since_case_start'],
        how='left'
    )
        .filter(
        (F.col('milestone_seconds_since_case_start') <= F.col('seconds_since_case_start')) |
        (F.col('seconds_since_case_start') == 0) |
        (F.col('total_milestones').isNull())
    )
        .withColumn(
        'seconds_to_closest_milestone_features',
        F.col('seconds_since_case_start') - F.col('milestone_seconds_since_case_start')
    )
        .withColumn(
        'rank_seconds_to_milestone_features',
        F.row_number().over(
            Window
                .partitionBy('reference_id', 'seconds_since_case_start')
                .orderBy("seconds_to_closest_milestone_features")
        )
    )
        .filter(
        F.col('rank_seconds_to_milestone_features') == 1
    )

)

pivoted_milestone_ids_features.show()

# pivoted_milestone_ids_features.filter(F.col('seconds_to_closest_milestone_features') != 0).show()
# pivoted_milestone_ids_features.sample(False, 0.001).show()

spark_write.parquet(
    df=pivoted_milestone_ids_features,
    path=data_dir.make_feature_path('pivoted_milestone_ids'),
    n_partitions=10
)

pivoted_milestone_ids_features = spark_read.parquet(
    data_dir.make_feature_path('pivoted_milestone_ids')
)

pivoted_milestone_ids_features.show()
print(f'pivoted_milestone_ids_features has {pivoted_milestone_ids_features.count()} rows')

#
#
# import lightgbm as lgb
# import numpy as np
# import pandas as pd
#
# from config.data_paths import data_dir
# # from config.pandas_output import *
#
# np.random.seed(10)
# baseline_model_file = pd.read_parquet(data_dir.make_feature_path('milestone_ids'))
# baseline_model_file['in_validation'] = np.random.choice(
#     a=[1, 0],
#     size=baseline_model_file.shape[0],
#     p=[0.75, 0.25]
# )
#
# baseline_model_file.head(20)
#
# non_feature_columns = [
# 'reference_id'
#
# ]  ## + ['product_' + str(idx+1) for idx in range(21)]
# response = 'inverse_time_to_next_escalation'
#
# feature_columns = (
#     baseline_model_file
#         .columns
#         .drop(non_feature_columns)
#         .drop(response)
# )
#
# train = (
#     baseline_model_file.loc[
#         (baseline_model_file['in_train'] == 1) &
#         (baseline_model_file['in_validation'] == 0)
#         ]
# )
# validation = (
#     baseline_model_file.loc[
#         (baseline_model_file['in_train'] == 1) &
#         (baseline_model_file['in_validation'] == 1)
#         ]
# )
#
# test = (
#     baseline_model_file.loc[
#         (baseline_model_file['in_train'] == 0)
#     ]
# )
#
# categorical_features = [
#
# ]
#
# print('lgb dataset train')
# train_set = lgb.Dataset(
#     data=train[feature_columns],
#     label=train[response],
#     categorical_feature=categorical_features
#     # weight=train['weight']
# )
# print('lgb dataset test')
# val_set = lgb.Dataset(
#     data=validation[feature_columns],
#     label=validation[response],
#     categorical_feature=categorical_features,
#     reference=train_set
#     # weight=validation['weight']
# )
#
# # del train
#
# # define random hyperparammeters
# params = {
#     'boosting_type': 'gbdt',
#     'metric': ['rmse'],  # , 'rmse'],
#     # 'first_metric_only': True,
#     'objective': 'rmse',
#     'n_jobs': -1,
#     'seed': 10,
#     'learning_rate': 0.055,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 10,
#     'colsample_bytree': 1,
#     'cat_smooth': 1,
#     'cat_l2': 1,
#     'min_data_per_group': 10
#     # 'force_row_wise': True,
#     # 'max_depth': 3,
#     # 'min_data_in_leaf': 80,
#     # 'num_leaves': 200,
#     # 'early_stopping_rounds': 50,
#     # "colsample_bytree": 0.75,
#     # "lambda": 0.1,
#     # "alpha": 0.1
# }
#
# model = lgb.train(
#     params=params,
#     train_set=train_set,
#     num_boost_round=2000,
#     early_stopping_rounds=30,
#     valid_sets=[train_set, val_set],
#     verbose_eval=1,
#     # feval=wrmsse
# )
#
#
#
# from source.utils.trained_model_information import TrainedModelInformation
#
# model_information = TrainedModelInformation(
#     model=model,
#     feature_columns=list(feature_columns)
# )
#
# model_information.get_feature_importance()
#
# from source.utils.ml_tools.predicted_vs_observed import PredictedAndObservedRegression
#
# predictions_validation = model.predict(
#     validation[feature_columns]
# )
# validation['pred'] = predictions_validation
# validation['pred'].mean()
# validation
#
# pvo_validation = PredictedAndObservedRegression(
#     predicted=predictions_validation,
#     observed=validation[response],
#     quantiles=100,
#     index=validation.index
# )
#
# pvo_validation.plot_predicted_and_observed()
#
# from sklearn.metrics import r2_score
#
# r2_score(
#     y_true=validation[response],
#     y_pred=predictions_validation
# )
