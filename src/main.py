import json
from operator import add
import re
from pyspark import SparkConf, SQLContext
from pyspark.sql import SparkSession, SQLContext


def init_spark():
    spark = SparkSession.builder.appName("wireshark").master("spark://172.20.20.20:7077").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


if __name__ == '__main__':
    spark, sc = init_spark()
