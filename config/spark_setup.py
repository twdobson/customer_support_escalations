import os

import findspark
from pyspark.sql import SparkSession


def launch_spark(app_name: str = 'app', platform: str = "windows"):
    """

    :param app_name:
    :param platform: The palform for which the code is being run off. This includes: "windows", "mac", "cluster"
    :return:
    """

    if platform == "windows":
        os.environ['HADOOP_HOME'] = 'C:\\Users\\Graduate\\PycharmProjects\\projection\\hadoop_home'
    elif platform == "mac":
        os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home"
        findspark.init("/usr/local/Cellar/apache-spark/3.0.0/libexec/")
    elif platform == "cluster":
        pass
    else:
        pass

    spark = (
        SparkSession
            .builder
            .config("spark.executor.memory", '10g')
            .config("spark.driver.memory", '10g')
            .config("spark.sql.crossJoin.enabled", "true")
            # .config('spark.sql.codegen.wholeStage', 'false')
            # .appName(app_name)
            .getOrCreate()
    )

    return spark



import os

