import os

from config.spark_setup import launch_spark

FULL_RUN: bool = True
WORKING_DIRECTORY: str = r"rC:\Users\Graduate\PycharmProjects\unique_offers"
DATA_FOLDER: str = "data" if FULL_RUN else os.path.join('data', 'sample')
PLATFORM: str = "mac"

spark = launch_spark(
    app_name="insurance_takeup",
    platform=PLATFORM
)
