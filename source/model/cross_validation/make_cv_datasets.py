# from config.env import *
import gc
import os

import lightgbm as lgb

from config.data_paths import data_dir
from config.pandas_output import *
from source.utils.reader_writers.reader_writers import write_object, load_object

# feature_columns_top = list(pd.read_csv(data_dir.make_processed_path('top_features.csv'))['feature'])
response = 'response'

print('reading in model file')
train_validate_file = pd.read_parquet(data_dir.make_processed_path('model_files', 'train_validate_file'))
print(train_validate_file.dtypes)
print(train_validate_file.head())
non_feature_columns = [
    'reference_id'
]

# train_validate_file['q'] = round(train_validate_file['reference_id'].rank(pct=True),3)
# train_validate_file.groupby('q')[response].mean().plot()

feature_columns = (
    train_validate_file
        .columns
        .drop(non_feature_columns)
        .drop(response)
)

write_object(
    path=data_dir.make_processed_path('parameters', 'feature_columns'),
    py_object=feature_columns
)

validation_fold_names = load_object(data_dir.make_processed_path('parameters', 'validation_fold_names'))
k_folds = load_object(data_dir.make_processed_path('parameters', 'k_folds'))

for fold_name in validation_fold_names:
    print(f'collecting and saving fold indices for fold {fold_name}')
    cv_indices = {
        idx: train_validate_file[fold_name][train_validate_file[fold_name] == idx].index
        for idx
        in range(0, k_folds, 1)
    }

    write_object(
        path=data_dir.make_processed_path('parameters', 'cv_indices', fold_name),
        py_object=cv_indices
    )

categorical_features = [
    # 'vendor_id'
]

print('preparing lgb dataset')
train_validate_dataset = lgb.Dataset(
    data=train_validate_file[feature_columns],
    label=train_validate_file[response],
    categorical_feature=categorical_features,
    # weight=train_validate_file['is_purchased_small_weight']
)

del train_validate_file
gc.collect()

print('saving model file binary')

try:
    os.remove(data_dir.make_processed_path('models', 'datasets', 'train_validate_dataset'))
except:
    pass

train_validate_dataset.save_binary(
    filename=data_dir.make_processed_path('models', 'datasets', 'train_validate_dataset')
)

del train_validate_dataset
gc.collect()
