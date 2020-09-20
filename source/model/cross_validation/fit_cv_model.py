import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

PD_MAX_ROWS = 500
PD_MAX_COLUMNS = 5100
PD_CONSOLE_WIDTH = 2000
PD_MAX_COLWIDTH = 1000

pd.options.display.max_rows = PD_MAX_ROWS
pd.options.display.max_columns = PD_MAX_COLUMNS
pd.options.display.width = PD_CONSOLE_WIDTH
pd.options.display.max_colwidth = PD_MAX_COLWIDTH

from config.data_paths import data_dir
from config.pandas_output import *
from source.utils.reader_writers.reader_writers import load_object, write_object
from source.utils.trained_model_information import TrainedModelInformation

response = 'response'
MODEL_VERSION = "2_baseline_recreated"
MODEL_KEY_COLUMN = 'reference_id'

indices = load_object(path=data_dir.make_processed_path('parameters', 'cv_indices', 'cross_validation_fold_0'))
feature_columns = load_object(path=data_dir.make_processed_path('parameters', 'feature_columns'))
print(len(feature_columns))

params = {
    'boosting_type': 'gbdt',
    'metric': ['rmse'],  # , 'rmse'],
    # 'first_metric_only': True,
    'objective': 'rmse',
    'n_jobs': -1,
    'seed': 10,
    'learning_rate': 0.01,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'colsample_bytree': 0.85,
    # "lambda": 100
    # 'sigmoid': 0.9,
    # "boost_from_average": False,
    # 'force_row_wise': True,
    # 'max_depth': 4,
    # 'min_data_in_leaf': 5,
    # 'num_leaves': 30,
    # 'extra_trees': True,
    # "scale_pos_weight" : 3,
    # "is_unbalance": True,
    # 'early_stopping_rounds': 50,
    # "colsample_bytree": 1,
    # "alpha": 25.5,
    # "lambda": 25.5,
    # 'cat_smooth': 5,
    # 'cat_l2': 2.5,
    # 'extra_seed': 99

}

for validation_fold in range(0, 5, 1):
    print("***************************************************************************************")
    print(f"*********************** validation fold: {validation_fold}****************************")
    print("***************************************************************************************")
    print('reading in train validation file')
    train_validate_file = lgb.Dataset(data_dir.make_processed_path('models', 'datasets', 'train_validate_dataset'))

    train_indices = [idx for fold in range(0, 5, 1) if fold != validation_fold for idx in indices.get(fold)]
    validation_indices = indices.get(validation_fold)
    print(f'total train indices {len(train_indices) + len(validation_indices)}')

    print('preparing modelling datasets')
    train = train_validate_file.subset(train_indices)
    validation = train_validate_file.subset(validation_indices)
    # print(f"number of features in training data {train_validate_file.num_feature()}")

    del train_validate_file
    gc.collect()

    model = lgb.train(
        params=params,
        train_set=train,
        num_boost_round=10000,
        early_stopping_rounds=25,
        valid_sets=[train, validation],
        verbose_eval=1
    )

    write_object(
        path=data_dir.make_processed_path(
            'model',
            'fit_models',
            f'model_{MODEL_VERSION}',
            f'{validation_fold}_cv'
        ),
        py_object=model
    )

    model_information = TrainedModelInformation(
        model=model,
        feature_columns=feature_columns
    )

    print(model_information.get_feature_importance().head(300))

# ['average_length_of_note_description_null_for_6763', 'proportion_is_milestone_note_6763']

from sklearn.metrics import r2_score

r2_scores = []
validation_predictions = []
validation_responses = []
for fold in range(0, 5, 1):
    model = load_object(path=data_dir.make_processed_path(
        'model',
        'fit_models',
        f'model_{MODEL_VERSION}',
        f'{fold}_cv'
    ))
    print('predicting test set')
    validation_indices = indices.get(fold)
    train_validate_file = pd.read_parquet(data_dir.make_processed_path("model_files", 'train_validate_file'))
    predictions = model.predict(train_validate_file.iloc[validation_indices, :][feature_columns])

    validation_predictions.append(predictions)
    validation_responses.append(train_validate_file.iloc[validation_indices, :][response])

    r2 = r2_score(
        y_true=train_validate_file.iloc[validation_indices, :][response],
        y_pred=predictions
    )
    print(r2)
    r2_scores.append(r2)

print(sum(r2_scores) / len(r2_scores))
sum(validation_predictions[2]) / len(validation_predictions[2])

from source.utils.ml_tools.predicted_vs_observed import PredictedAndObservedRegression

pvo = PredictedAndObservedRegression(
    predicted=validation_predictions[0],
    observed=validation_responses[0].astype('float'),
    quantiles=50,
    index=validation_responses[0].index
)
pvo.plot_predicted_and_observed()
pvo.plot_pvo()

for fold in range(0, 5, 1):
    model = load_object(path=data_dir.make_processed_path(
        'model',
        'fit_models',
        f'model_{MODEL_VERSION}',
        f'{fold}_cv'
    ))
    print('predicting test set')
    test = pd.read_parquet(data_dir.make_processed_path("model_files", 'test_file'))
    predictions_test = model.predict(test[feature_columns])
    test['prediction'] = predictions_test
    print(f"mean of predictions {np.mean(predictions_test)}")
    sample_submission = pd.read_csv(data_dir.make_raw_path('IBI_test_cases_no_target.csv'))

    sample_submission.head()

    submission = pd.merge(
        sample_submission,
        test[['reference_id', 'prediction']],
        left_on=['REFERENCEID'],
        right_on=['reference_id']
    )
    submission['INV_TIME_TO_NEXT_ESCALATION'] = submission['prediction']
    submission.drop(
        columns=['reference_id', 'prediction'],
        inplace=True
    )

    print('saving test predictions to disk')
    submission.to_csv(
        path_or_buf=data_dir.make_processed_path(
            'submissions',
            'cv',
            f'test_version_{MODEL_VERSION}_fold_{fold}.csv'),
        index=False
    )

    print(f'finished fold {fold}')

    gc.collect()

# import seaborn as sns
#
# col = 'seconds_since_case_start'
# sns.distplot(train_validate_file[response])
# sns.distplot(predictions_test)
#
# import seaborn as sns
#
# col = 'seconds_since_case_start'
# sns.distplot(train_validate_file.loc[train_validate_file[col] < 10000, col])
# sns.distplot(test.loc[test[col] < 10000, col])

test_predictions_all_folds = [
    pd.read_csv(
        data_dir.make_processed_path(
            'submissions',
            'cv',
            f'test_version_{MODEL_VERSION}_fold_{fold}.csv')
    )
    for fold in range(0, 5, 1)
]

test_predictions_all_folds = pd.concat(test_predictions_all_folds, axis=0)
test_predictions_all_folds.groupby("REFERENCEID")['INV_TIME_TO_NEXT_ESCALATION'].mean().sort_values()
test = test_predictions_all_folds.groupby("REFERENCEID", as_index=False)[['INV_TIME_TO_NEXT_ESCALATION']].mean()
sample_submission = pd.read_csv(data_dir.make_raw_path('IBI_test_cases_no_target.csv'))

final_submission = pd.merge(
    sample_submission.drop(columns=['INV_TIME_TO_NEXT_ESCALATION']),
    test,
    on="REFERENCEID",
    how='inner'
)

final_submission.loc[final_submission['INV_TIME_TO_NEXT_ESCALATION'] < 0, 'INV_TIME_TO_NEXT_ESCALATION'] = 0
# 'REFERENCEID', 'SECONDS_SINCE_CASE_START',
final_submission[['INV_TIME_TO_NEXT_ESCALATION']].to_csv(
    path_or_buf=data_dir.make_processed_path(
        'submissions',
        'cv',
        f'test_version_{MODEL_VERSION}_combined_cv.csv'),
    index=False,
    header=False
)

final_submission_input = pd.read_csv(
    data_dir.make_processed_path(
        'submissions',
        'cv',
        f'test_version_{MODEL_VERSION}_combined_cv.csv')
)

# sns.distplot(final_submission_input['INV_TIME_TO_NEXT_ESCALATION'])
# train_validate_file = pd.read_parquet(data_dir.make_processed_path('model_files', 'train_validate_file'))
# sns.distplot(train_validate_file.loc[train_validate_file[response] < 0.08, response])
#
# train_validate_file.loc[:, response].mean()
