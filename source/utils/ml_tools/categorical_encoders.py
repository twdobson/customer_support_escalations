from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F


def label_encode_categorical(df: DataFrame, categorical_column: str):
    return (
        df
            .groupby(
            categorical_column
        )
            .agg(
            F.count("*").alias('group_count')
        )
            .withColumn(
            f'label_encoded_{categorical_column}',
            F.row_number().over(
                Window
                    .partitionBy()
                    .orderBy(F.desc("group_count"))
            )
        )
        .drop('group_count')
    )


def label_encode_categorical_inplace(df: DataFrame, categorical_column: str):
    return (
        df
            .join(
            label_encode_categorical(
                df=df,
                categorical_column=categorical_column
            ),
            on=[categorical_column],
            how='left'
        )
    )

