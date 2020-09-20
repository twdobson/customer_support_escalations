from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F


def label_encode_categorical(df: DataFrame, categorical_column: str) -> DataFrame:
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


def label_encode_categorical_inplace(df: DataFrame, categorical_column: str) -> DataFrame:
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


def one_hot_encode_categorical(
        df: DataFrame,
        categorical_column: str,
        values_to_one_hot_encode: List[str],
        return_column_names: bool = True
) -> DataFrame:
    for value in values_to_one_hot_encode:
        df = (
            df
                .withColumn(
                f'one_hot_encoded_{categorical_column}_is_{value}',
                F.when(
                    F.col(categorical_column) == value, 1
                ).otherwise(
                    0
                )
            )
        )

    one_hot_encoded_column_names = [
        f'one_hot_encoded_{categorical_column}_is_{value}'
        for value
        in values_to_one_hot_encode
    ]
    if return_column_names:
        return (df, one_hot_encoded_column_names)
    else:
        return df


def encode_categorical_using_mean_response_rate(
        df: DataFrame,
        categorical_column: str,
        response_column: str
) -> DataFrame:
    return (
        df
            .groupBy(
            categorical_column
        )
            .agg(
            F.avg(response_column).alias(f'mean_response_rate_{categorical_column}')
        )
    )


def encode_categorical_using_mean_response_rate_inplace(
        df: DataFrame, categorical_column: str,
        response_column: str
) -> DataFrame:
    return (
        df
            .join(
            encode_categorical_using_mean_response_rate(
                df=df,
                categorical_column=categorical_column,
                response_column=response_column
            ),
            on=[categorical_column],
            how='left'
        )
    )
