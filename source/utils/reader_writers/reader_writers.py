import time

import pickle
import os
from source.utils.pyspark.t_data_frame import TDataFrame


def make_path(path):
    if os.path.exists(os.path.dirname(path)):
        pass
    else:
        os.makedirs(os.path.dirname(path))


def write_object(path: str, py_object):
    make_path(path=path)
    print(f"writing pickled object to {path}")
    with open(path, 'wb') as writer:
        pickle.dump(py_object, writer)


def load_object(path: str):
    print(f"reading picked object from {path}")
    with open(path, 'rb') as py_object:
        py_object = pickle.load(py_object)

    return py_object


def read_logger(reader):
    def log(self, path: str):
        print(f"reading from {path}")

        return reader(self, path)

    return log


def write_logger(reader):
    write_start = time.time()

    def log(self, df, path: str, n_partitions: int):
        print(f"writing to {path}")
        print(f"with {n_partitions} partitions")

        return reader(self, df, path, n_partitions)

    print(f"{write_start - time.time()}s to write file")

    return log


class SparkRead:

    def __init__(self, spark):
        self.spark = spark

    @read_logger
    def csv(self, path: str) -> TDataFrame:
        return (
            TDataFrame(
                self
                    .spark
                    .read
                    .option('header', 'true')
                .option('inferSchema', 'true')
                    .csv(path)
            )
        )

    @read_logger
    def pipe(self, path: str) -> TDataFrame:
        return (
            TDataFrame(
                self
                    .spark
                    .read
                    .option("delimiter", "|")
                    .option('header', 'true')
                    .csv(path)
            )
        )

    @read_logger
    def parquet(self, path: str) -> TDataFrame:
        return (
            TDataFrame(
                self
                    .spark
                    .read
                    .parquet(path)
            )
        )


class SparkWrite:

    def csv(self, df: TDataFrame, path: str, n_partitions: int = 1):
        print(f"writing to {path}")
        print(f"with {n_partitions} partitions")
        return (
            df
                .repartition(n_partitions)
                .write
                .mode("overwrite")
                .option("header", "true")
                .csv(path)
        )

    def parquet(self, df: TDataFrame, path: str, n_partitions: int):
        print(f"writing to {path}")
        print(f"with {n_partitions} partitions")
        return (
            df
                .repartition(n_partitions)
                .write
                .mode("overwrite")
                .parquet(path)
        )
