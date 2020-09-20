from pyspark.sql import functions as F

from config.data_paths import data_dir
from config.env import *
from source.utils.reader_writers.reader_writers import (
    SparkRead,
    SparkWrite,
    write_object
)

spark_read = SparkRead(spark=spark)
spark_write = SparkWrite()

# ******************************************************************
# ******************** COMMENTS ***********************************
# *******************************************************************


raw_comments = spark_read.csv(
    data_dir.make_raw_path("IBI_case_comments_anonymized.csv")
)
raw_comments.show(truncate=True)
raw_comments.printSchema()

raw_comments.create_unique_value_summary(columns=raw_comments.columns).orderBy('unique_count').show(truncate=False)
raw_comments.create_null_summary().show(truncate=False)

comments_column_map = {
    'REFERENCEID': 'reference_id',
    'SECONDS_SINCE_CASE_START': 'seconds_since_case_start',
    'NOTES': 'notes',
    'CREATED_BY': 'created_by',
    'GLOBALID': 'global_id',
    'COMMENT_TYPE': 'comment_type'
}

raw_comments.groupby('COMMENT_TYPE').agg(F.count('*')).show()

comments = (
    raw_comments
        .toDF(*[comments_column_map.get(col, col) for col in raw_comments.columns])
        .withColumn(
        'comment_type',
        F.lower(F.col('comment_type'))
    )
        .withColumn(
        'inverse_seconds_since_case_start',
        1 / F.col('seconds_since_case_start')
    )
)

comments.show()

spark_write.parquet(
    df=comments,
    path=data_dir.make_interim_path('comments'),
    n_partitions=10
)

# ******************************************************************
# ******************** METADATA ***********************************
# *******************************************************************

raw_metadata = spark_read.csv(
    path=data_dir.make_raw_path('IBI_case_metadata_anonymized.csv')
)
raw_metadata.show()


raw_metadata.create_unique_value_summary(columns=raw_metadata.columns).orderBy('unique_count').show(truncate=False)
raw_metadata.create_null_summary().show(truncate=False)

meta_data_column_map = {
    'REFERENCEID': 'reference_id',
    'SITECODE': 'site_code',
    'SITENAME': 'site_name',
    'CONTACTCOMPANY': 'contact_company',
    'COMPANYID': 'company_id',
    'COMPANYOID': 'company_o_id',
    'INITIALUSERGROUPID': 'initial_user_group_id',
    'INITIALUSERGROUPDESC': 'initial_user_group_desc',
    'MEDPROJECTID': 'med_project_id',
    'MEDPROJECTAREA': 'med_project_area',
    'MEDPROJOPENBY': 'med_project_open_by',
    'IWAYJIRAISSUEID': 'i_way_jira_issue_id',
    'NEWIWAYJIRAISSUEID': 'new_i_way_jira_issue_id',
    "PREMIUMCODE": 'premium_code',
    'PRIMARYPRODUCTFAMILYID': 'primary_product_family_id',
    'PRIMARYPRODUCTFAMILYDESC': 'primary_product_family_desc',
    'PRIMARYPRODUCTID': 'primary_product_id',
    'PRIMARYPRODUCTDESC': 'primary_product_desc',
    'PRIMARYRELEASEID': 'primary_release_id',
    'PRIMARYRELEASEDESC': 'primary_release_desc',
    'PRIMARYPRODUCTAREAID': 'primary_product_area_id',
    'PRIMARYPRODUCTAREADESC': 'primary_product_area_desc',
    'PRIMARYPRODUCTOSFAMILYID': 'primary_product_os_family_id',
    'PRIMARYPRODUCTOSFAMILYDESC': 'primary_product_os_family_desc',
    'PRIMARYPRODUCTOSPLATFORMID': 'primary_product_os_platform_id',
    'PRIMARYPRODUCTOSPLATFORMDESC': 'primary_product_os_platform_desc',
    'CONTACTMETHODFLAG': 'contact_method_flag',
    'CUSTOMERFIRST': 'customer_first',
    'CUSTOMERMIDDLE': 'customer_middle',
    'CUSTOMERLAST': 'customer_last',
    'CUSTOMERNAME': 'customer_name',
    'CONTACTPHONE': 'customer_phone',
    'CONTACTMOBILEPHONE': 'contact_mobile_phone',
    'CONTACTEMAIL': 'contact_email',
    'IBICUSTOMERFIRST': 'ibi_customer_first',
    'IBICUSTOMERMI': 'ibi_customer_i',
    'IBICUSTOMERLAST': 'ibi_customer_last',
    'IBICUSTOMERNAME': 'ibi_customer_name',
    'IBIPHONE': 'ibi_phone',
    'IBIMOBILE': 'ibi_mobile',
    'IBIEMAIL': 'ibi_email',
    'CUSTOMERLABEL': 'customer_label',
    'BRANCHCODE': 'branch_code',
    'COUNTRY': 'country',
    'AGENT_ID': 'agent_id',
    'AGENT_NAME': 'agent_name',
    'ASM_NAME': 'asm_name',
    'SITECOMPANYNAME': 'site_company_name',
    'SITECOMPANY_OID': 'site_company_o_id',
    'SITECOMPANYID': 'site_company_id',
    'PRIMARYPRODUCTVERSION': 'primary_product_version',
    'CLOUD': 'cloud',
    'CASENUM': 'case_number',
    'PROJNUM': 'project_number',
    'IWAYJIRA': 'i_way_jira',
    'PNOTARGET': 'pno_target',
    'PNOCRITICALUSER': 'pno_critical_user',
    'ISPREMIUM': 'is_premium',
    'CUSTOMER_NAME': 'customer_name',
    'CUSTOMER_LABEL': 'customer_label',
    'GLOBAL_ID': 'global_id',
    'CUSTOMER_PHASE': 'customer_phase',
    'P1PHONE': 'p1_phone',
    'SITEINSTALLYEAR': 'site_install_year'
}

metadata = (
    raw_metadata
        .toDF(*[meta_data_column_map.get(col, col) for col in raw_metadata.columns])
        .withColumn(
        'is_premium',
        F.col('is_premium').cast('int')
    )
        .withColumn(
        'site_install_year',
        F.col('site_install_year').cast('int')
    )
        .withColumn(
        'premium_code',
        F.col('premium_code').cast('int')
    )
        .drop(
        'customer_name',
        'customer_label'
    )
)

metadata.show()

spark_write.parquet(
    df=metadata,
    path=data_dir.make_interim_path('metadata'),
    n_partitions=10
)


# ******************************************************************
# ******************** MILESTONES ***********************************
# *******************************************************************

raw_milestones = spark_read.csv(
    path=data_dir.make_raw_path('IBI_case_milestones_anonymized.csv')
)

milestones_column_map = {
    'REFERENCEID': 'reference_id',
    'SECONDS_SINCE_CASE_START': 'seconds_since_case_start',
    'MILESTONEID': 'milestone_id',
    'MILESTONEDESCRIPTION': 'milestone_description',
    'NOTEDESCRIPTION': 'note_description',
    'CREATED_BY': 'created_by',
    'UPDATED_BY': 'updated_by'
}

milestones = (
    raw_milestones
        .toDF(*[milestones_column_map.get(col, col) for col in raw_milestones.columns])
        .withColumn(
        'inverse_seconds_since_case_start',
        1 / F.col('seconds_since_case_start')
    )
)
milestones.show()

spark_write.parquet(
    df=milestones,
    path=data_dir.make_interim_path('milestones'),
    n_partitions=10
)

# ******************************************************************

raw_case_status_history = spark_read.csv(
    path=data_dir.make_raw_path('IBI_case_status_history_v2.csv'),
)
raw_case_status_history.show(truncate=False)
raw_case_status_history.filter(F.col('INV_TIME_TO_NEXT_ESCALATION') != 0).show()

case_status_history_column_map = {
    'REFERENCEID': 'reference_id',
    'SECONDS_SINCE_CASE_START': 'seconds_since_case_start',
    'SEVERITY': 'severity',
    'ISESCALATE': 'is_escalate',
    'INV_TIME_TO_NEXT_ESCALATION': 'inverse_time_to_next_escalation'
}

case_status_history = (
    raw_case_status_history
        .toDF(*[case_status_history_column_map.get(col, col) for col in raw_case_status_history.columns])
)
case_status_history.show()

spark_write.parquet(
    df=case_status_history,
    path=data_dir.make_interim_path('case_status_history'),
    n_partitions=10
)

# start of important EDA
#
# sns.distplot(
#     case_status_history
#         .withColumn(
#         'TIME_TO_NEXT_ESCALATION',
#         1 / F.col('INV_TIME_TO_NEXT_ESCALATION')
#     )
#         # .filter(F.col('INV_TIME_TO_NEXT_ESCALATION')!=0)
#         .toPandas()['TIME_TO_NEXT_ESCALATION']
# )

raw_test = spark_read.csv(
    data_dir.make_raw_path('IBI_test_cases_no_target.csv')
)
raw_test.show()
raw_test.count()  # 12724
raw_test.select('REFERENCEID').distinct().count()  # 12724

test = (
    raw_test
        .toDF(*[case_status_history_column_map.get(col, col) for col in raw_test.columns])
)

spark_write.parquet(
    df=test,
    path=data_dir.make_interim_path('test'),
    n_partitions=10
)

interim_response_file = (
    case_status_history
        .select(
        'reference_id',
        'seconds_since_case_start',
        'inverse_time_to_next_escalation'
    )
        .withColumn(
        'in_train',
        F.when(
            F.col('inverse_time_to_next_escalation').isNotNull(), 1
        ).otherwise(
            0
        )
    )
        .filter(
        F.col('in_train') == 1
    )
        .select(
        'reference_id',
        'seconds_since_case_start',
        'inverse_time_to_next_escalation',
        'in_train'
    )
        .union(
        test
            .withColumn(
            'in_train',
            F.lit(0)
        )
    )
)

k_folds = 5

validation_folds = (
    interim_response_file
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
response_file = (
    interim_response_file
        .join(
        validation_folds,
        on=['reference_id'],
        how='inner'
    )
        .withColumn(
        'inverse_time_to_next_escalation',
        F.col('inverse_time_to_next_escalation').cast('double')
    )
)

spark_write.parquet(
    df=response_file,
    path=data_dir.make_interim_path('response_file'),
    n_partitions=10
)

write_object(
    path=data_dir.make_processed_path('parameters', 'validation_fold_names'),
    py_object=['cross_validation_fold_0', 'cross_validation_fold_1']
)
write_object(
    path=data_dir.make_processed_path('parameters', 'k_folds'),
    py_object=5
)

# ******************************************************************
# ******************** ID_TO_LEMMA ***********************************
# *******************************************************************

id_to_lemma = spark_read.csv(
    path=data_dir.make_raw_path('id_to_lemma_public_translations.csv')
)

id_to_lemma.show()

spark_write.parquet(
    df=id_to_lemma,
    path=data_dir.make_interim_path('id_to_lemma'),
    n_partitions=10
)

# ******************************************************************
# ******************** DICTIONARY ***********************************
# *******************************************************************

raw_dictionary = spark_read.csv(
    path=data_dir.make_raw_path('challenge_dictionary_info.csv')
)
raw_dictionary.show()
raw_dictionary.count()

dictionary_column_map = {
    'id': 'word_id',
    'count': 'word_count',
    'ner_type': 'ner_type',
    'pos': 'pos',
    'in_dictionary': 'in_dictionary'
}
raw_dictionary.groupby('in_dictionary').agg(F.count("*")).show()

dictionary = (
    raw_dictionary
        .toDF(*[dictionary_column_map.get(col, col) for col in raw_dictionary.columns])
        .withColumn(
        'is_in_dictionary',
        F.when(
            F.col('in_dictionary') == 'true', 1
        ).when(
            F.col('in_dictionary') == 'false', 0
        ).otherwise(
            None
        )
    )
        .drop(
        'in_dictionary'
    )
)

dictionary.show()

spark_write.parquet(
    df=dictionary,
    path=data_dir.make_interim_path('dictionary'),
    n_partitions=10
)
