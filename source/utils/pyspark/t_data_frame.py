from __future__ import annotations
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import (
    List,
    Iterable
)



class TDataFrame(DataFrame):

    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def create_null_summary(
            self,
            columns: List[str] = None,
            groupby_columns: List[str] = None,
            table_name: str = None
    ) -> TDataFrame:
        groupby_columns = groupby_columns or []
        columns = columns or [col for col in self.columns if col not in groupby_columns]

        aggs = [F.count(F.when(F.col(col).isNull(), col)).alias(col) for col in columns]
        aggs.append(F.count("*").alias("count_in_group"))

        null_summary = TDataFrame(
            self
                .groupby(*groupby_columns)
                .agg(*aggs)
        )

        null_summary = null_summary.melt(
            id_vars=groupby_columns + ['count_in_group'],
            value_vars=columns,
            var_name='column',
            value_name='null_count'
        )

        null_summary = (
            null_summary
                .withColumn(
                'percent_of_group_null',
                F.col("null_count") / F.col("count_in_group")
            )
        )

        if table_name is not None:
            null_summary = (
                null_summary
                    .withColumn(
                    'table_name',
                    F.lit(table_name)
                )
            )

        return null_summary

    def melt(
            self,
            value_vars: Iterable[str],
            id_vars: Iterable[str] = None,
            var_name: str = "variable",
            value_name: str = "value"
    ) -> TDataFrame:
        """

        :param self:
        :param value_vars:
        :param id_vars:
        :param var_name:
        :param value_name:
        :return:

        Convert :class:`DataFrame` from wide to long format.

        """
        id_vars = id_vars if id_vars is not None else []
        # Create array<struct<variable: str, value: ...>>
        variable_name_with_column_values = F.array(*(
            F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name))
            for c in value_vars))

        # Add to the DataFrame and explode
        exploded_vars_and_vals = self.withColumn(
            "variable_name_with_column_values",
            F.explode(variable_name_with_column_values)
        )

        cols = id_vars + [
            F.col("variable_name_with_column_values")[x].alias(x) for x in [var_name, value_name]
        ]
        return exploded_vars_and_vals.select(*cols)

    def create_distribution(
            self,
            groupby_columns: List[str] = None,
            numeric_column: List[str] = None,
            include_count_distribution: bool = True
    ) -> TDataFrame:
        """

        :param self:
        :param groupby_columns:
        :param numeric_column:
        :param include_count_distribution:
        :return:
        """
        aggs = []
        groupby_columns = groupby_columns or []

        if include_count_distribution:
            aggs.append(F.count("*").alias('count'))
        if numeric_column is not None:
            aggs.append(F.sum(numeric_column).alias(f"sum_{numeric_column}"))

        distribution = (

            self
                .groupby(*groupby_columns)
                .agg(*aggs)

        )

        if include_count_distribution:
            row_count = self.count()
            distribution = (
                distribution
                    .withColumn(
                    'count_percentage',
                    F.col('count') / F.lit(row_count)
                )
            )

        if numeric_column is not None:
            numeric_column_total = (
                self
                    .agg(
                    F.sum(numeric_column).alias(f"{numeric_column}_total")
                )
                    .collect()[0][0]
            )

            distribution = (
                distribution
                    .withColumn(
                    f"{numeric_column}_percentage",
                    F.col(f"sum_{numeric_column}") / F.lit(numeric_column_total)
                )
            )

        return distribution

    def create_long_distribution(
            self,
            categorical_col: str,
            numeric_column: str = None,
            include_count_distribution: bool = True,
            table_name: str = None
    ) -> TDataFrame:
        """

        :param self:
        :param categorical_col:
        :param numeric_column:
        :param include_count_distribution:
        :param table_name:
        :return:
        """
        distribution = self.create_distribution(
            groupby_columns=[categorical_col],
            numeric_column=numeric_column,
            include_count_distribution=include_count_distribution
        )

        distribution = self.melt(
            id_vars=[categorical_col],
            value_vars=[column for column in distribution.columns if column != categorical_col],
            var_name='statistic',
            value_name='statistic_value'
        )

        distribution = (
            distribution
                .withColumn(
                'categorical',
                F.lit(categorical_col)
            )
                .withColumnRenamed(
                categorical_col,
                'categorical_value'
            )
        )

        if table_name is not None:
            distribution = (
                distribution
                    .withColumn(
                    'table_name',
                    F.lit(table_name)
                )
            )

        return distribution

    def create_unique_value_summary(
            self,
            columns: List[str] = None,
            groupby_columns: List[str] = None,
            include_percent: bool = False
    ) -> TDataFrame:
        """

        :param self:
        :param columns:
        :param groupby_columns:
        :param include_percent:
        :return:
        """
        groupby_columns = groupby_columns or []
        aggs = [F.countDistinct(col).alias(f"count_unique_{col}") for col in columns]
        agg_names = [f"count_unique_{col}" for col in columns]

        if include_percent:
            aggs.extend(
                [(F.countDistinct(col) / F.lit(F.sum(F.when(self[col].isNotNull(), 1).otherwise(0)))).alias(
                    f"percent_unique_{col}")
                    for col
                    in columns
                ]
            )
            agg_names.extend([f"percent_unique_{col}" for col in columns])

        unique_value_summary = TDataFrame(
            self
                .groupby(*groupby_columns)
                .agg(*aggs)
        )

        unique_value_summary = unique_value_summary.melt(
            id_vars=groupby_columns,
            value_vars=agg_names,
            var_name='column',
            value_name='unique_count'

        )
        unique_value_summary = (
            unique_value_summary
                .withColumn(
                'type',
                F.when(F.col("column").contains("percent"), "percent").otherwise("count")
            )
        )

        return unique_value_summary

    def create_numeric_summary(
            self,
            numeric_columns: List[str],
            groupby_columns: List[str] = None,
            table_name: str = None
    ) -> TDataFrame:
        """

        :param self:
        :param numeric_columns:
        :param groupby_columns:
        :return:
        """
        groupby_columns = groupby_columns if groupby_columns is not None else []

        aggs = []
        agg_column_names = []

        for agg, agg_name in [
            (F.min, 'min'),
            (F.avg, 'average'),
            (F.stddev, 'stddev'),
            (F.max, 'max'),
            (F.min, 'max'),
            (F.sum, 'sum'),
            (F.count, 'count'),
        ]:
            for col in numeric_columns:
                aggs.append(agg(col).alias(f"{agg_name}_{col}"))
                agg_column_names.append(f"{agg_name}_{col}")

        numeric_summary = (
            self
                .groupby(*groupby_columns)
                .agg(*aggs)
        )

        numeric_summary = numeric_summary.melt(
            id_vars=groupby_columns,
            value_vars=agg_column_names,
            value_name="statistic",
            var_name="statistic_value"
        )

        if table_name is not None:
            numeric_summary = (
                numeric_summary
                    .withColumn(
                    'table_name',
                    F.lit(table_name)
                )
            )

        return numeric_summary


