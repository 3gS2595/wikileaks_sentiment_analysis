import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType
from nltk.sentiment import SentimentIntensityAnalyzer

get_sentimentUDF = udf(lambda x: get_sentiment(x), ArrayType(FloatType()))


def init_spark():
    spark = SparkSession.builder.appName('wireshark').master('spark://172.20.20.20:7077').getOrCreate()
    sc = spark.sparkContext
    return spark, sc


# analyzes string returns array of sentiment scores
def get_sentiment(x):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(x)
    return [
        sentiment_score.get('neg'),
        sentiment_score.get('neu'),
        sentiment_score.get('pos'),
        sentiment_score.get('compound')
    ]


def main():
    spark, sc = init_spark()
    dfdirty = spark \
        .read \
        .option('inferSchema', 'true') \
        .option('header', 'true') \
        .csv('data/wikileaked.csv')

    #
    dfclean = (
        dfdirty
        .select(
            sf.substring(sf.col('timestamp and message'), 2, 10).alias('date'),
            sf.split(sf.split(sf.col('timestamp and message'), "<").getItem(1), ">").getItem(0).alias('name'),
            sf.regexp_replace(sf.col('timestamp and message'),
                              r"\[(0?\d{4}[-]\d{2}[-]\d{2}[ ]\d{2}[:]\d{2}[:]\d{2}[]][ ][<].*?)>", "").alias('message')
        )
    )
    dfclean = dfclean.withColumn("message", sf.regexp_replace("message", r"0?[0-9]", ""))
    dfclean = dfclean.filter(sf.col('message').isNotNull())

    sdf = dfclean.withColumn("sid", get_sentimentUDF(col("message")))
    sdf = (
        sdf
        .select(
            'date',
            'name',
            "message",
            sf.col('sid')[0].alias("neg"),
            sf.col('sid')[1].alias("neu"),
            sf.col('sid')[2].alias("pos"),
            sf.col('sid')[3].alias("compound")
        )
    )
    sdf_filtered = sdf.filter((sf.col('neg') != 1.0) & (sf.col('neu') != 1.0) & (sf.col('pos') != 1.0))

    (
        sdf_filtered
        .agg(
            sf.avg(sf.col('neg')).alias('neg'),
            sf.avg(sf.col('neu')).alias('neu'),
            sf.avg(sf.col('pos')).alias('pos'),
            sf.avg(sf.col('compound')).alias('compound')
        ).show()
    )

    test = sdf_filtered.repartition(1)

    dftop = test.rdd.map(lambda r: (r.name, (1, r.message, r.neg, r.neu, r.pos, r.compound))) \
        .reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + "\n " + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4], x[5] + y[5])) \
        .map(lambda r: (r[0], r[1][0], r[1][1], r[1][2], r[1][3], r[1][4], r[1][5])) \
        .toDF(["name", "cnt", "message", "neg", "neu", "pos", "compound"])

    dftop = dftop.rdd.map(lambda r: (r.name, (r.cnt, r.message, r.neg, r.neu, r.pos, r.compound))) \
        .map(
        lambda r: (r[0], r[1][0], r[1][1], r[1][2] / r[1][0], r[1][3] / r[1][0], r[1][4] / r[1][0], r[1][5] / r[1][0])) \
        .toDF(["name", "cnt", "message", "neg", "neu", "pos", "compound"])

    pos = dftop.sort(col('compound').asc())
    neg = dftop.sort(col('compound').desc())

    pos.show(20)
    neg.show(20)
    print(neg.count())
    pos.write.csv('pos.csv')
    neg.write.csv('neg.csv')


if __name__ == '__main__':
    main()
