# The below code is based on the baseline models 'extract features summary
# extract_features_summary < - function(dt, ref_ids_escalated)

from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.utils.ml_tools.categorical_encoders import (
    label_encode_categorical_inplace,
    encode_categorical_using_mean_response_rate_inplace
)
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

metadata = spark_read.parquet(
    path=data_dir.make_interim_path('metadata')
)

ref_ids_escalated.show()

# *****************************************************************************************

# ******************************************************************************************
# ************************ ESCALATION INDICATORS ******************************************
# *******************************************************************************************

metadata_for_escalated_ids = (
    metadata
        .join(
        ref_ids_escalated,
        on=['reference_id'],
        how='inner'
    )
)

columns_for_metadata_is_escalated = [
    'site_code',
    'company_id',
    'initial_user_group_id',
    'med_project_area',
    'med_project_open_by',
    'primary_product_id',
    'site_company_id',
    'country',
    'branch_code'
]

tables_for_is_in_metadata = [
    (col, metadata_for_escalated_ids
     .select(col)
     .distinct()
     .withColumn(
        f'is_in_escalated_{col}',
        F.lit(1)
    ))
    for col
    in columns_for_metadata_is_escalated
]

metadata_is_in_escalated = (
    metadata
        .select(
        *
        ['reference_id'] +
        columns_for_metadata_is_escalated
    )
)

for (col, df) in tables_for_is_in_metadata:
    print(col)
    metadata_is_in_escalated = (
        metadata_is_in_escalated
            .join(
            df,
            on=[col],
            how='left'
        )
            .na
            .fill(
            value=0,
            subset=[f'is_in_escalated_{col}']
        )
            .drop(col)
    )

metadata_is_in_escalated.show()

# ******************************************************************************************
# ************************ LABEL ENCODING ******************************************
# *******************************************************************************************

columns_for_label_encoding = [
    "initial_user_group_id",
    "med_project_open_by",
    "med_project_area",
    "primary_product_id",
    "country",
    "branch_code",
    "agent_id"
]

metadata_label_encoded = (
    metadata
        .select(
        *
        ['reference_id'] +
        columns_for_label_encoding
    )
)

for col in columns_for_label_encoding:
    metadata_label_encoded = (
        label_encode_categorical_inplace(
            df=metadata_label_encoded,
            categorical_column=col
        )
            .drop(col)
    )

metadata_label_encoded.show()
metadata_label_encoded.count()  # 53635

# ******************************************************************************************
# ************************ IS DUMMY 6763 FEATURES ******************************************
# *******************************************************************************************

columns_for_is_dummy_metadata = [
    'med_project_open_by',
    'med_project_id',
    'med_project_area'
]

metadata_is_dummy_encoded = (
    metadata
        .select(
        *
        ['reference_id'] +
        columns_for_is_dummy_metadata
    )
)

for col in columns_for_is_dummy_metadata:
    metadata_is_dummy_encoded = (
        metadata_is_dummy_encoded
            .withColumn(
            f'is_dummy_6763_{col}',
            F.when(
                F.col(col) == 6763, 1
            ).otherwise(
                0
            )
        )
            .drop(col)
    )

# ******************************************************************************************
# ************************ ALL METADATA FEATURES ******************************************
# *******************************************************************************************
response_encoded_metadata_features = (
    metadata
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
    'initial_user_group_id',
    # 'med_project_id',
    'med_project_area',
    'primary_product_id',
    'branch_code',
    'country',
    'site_company_o_id',
    'agent_id'
]

mean_encoded_columns = [
    f'mean_response_rate_{col}'
    for col
    in columns_to_mean_response_rate_encode
]

for col in columns_to_mean_response_rate_encode:
    response_encoded_metadata_features = encode_categorical_using_mean_response_rate_inplace(
        df=response_encoded_metadata_features,
        categorical_column=col,
        response_column='response'
    )



metadata_features = (
    metadata
        .select(
        'reference_id',
        F.col('cloud').alias('is_cloud'),
        F.col('premium_code'),
        F.col('is_premium'),
        F.col('primary_product_family_id'),
        F.col('customer_phase')
    )
        .fillna(
        value=0,
        subset=['is_cloud']
    )
    .join(
        response_encoded_metadata_features
            .select(
            *['reference_id'] + mean_encoded_columns
        ),
        on=['reference_id'],
        how='inner'

    )
        .join(
        metadata_is_in_escalated,
        on=['reference_id'],
        how='inner'
    )
        .join(
        metadata_label_encoded,
        on=['reference_id'],
        how='inner'
    )
        .join(
        metadata_is_dummy_encoded,
        on=['reference_id'],
        how='inner'
    )
)


spark_write.parquet(
    df=metadata_features,
    path=data_dir.make_feature_path('metadata'),
    n_partitions=10
)

metadata_features = spark_read.parquet(
    path=data_dir.make_feature_path('metadata')
)

metadata_features.show()
print(f" metadata has {metadata_features.count()} rows")
print(f" metadata has {len(metadata_features.columns)} columns")
