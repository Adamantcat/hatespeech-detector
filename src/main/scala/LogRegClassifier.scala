/**
  * Created by Julia on 07.06.2017.
  */

import java.util
import scala.util.matching.Regex

import scala.collection.JavaConversions._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, Row, SQLContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StopWordsRemover, RegexTokenizer}

import org.apache.log4j.Logger
import org.apache.log4j.Level


object LogRegClassifier {

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val data = "C:\\Users\\Julia\\Documents\\BA-Thesis\\labeled_data.csv"

    //read as textfile, split at comma, reunite all fields larger than 6, tolowercase

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()
    import spark.implicits._

    var df = spark.read.option("header", "true").csv(data)

    //append truncated lines to previous tweet
    val iter = df.toLocalIterator()
    var current: Row = iter.next() // initialize with first element
    var list: util.List[Row] = new util.ArrayList[Row]()

    while (iter.hasNext) {
      val next = iter.next()
      if (!next.anyNull) {
        list.add(current)
        current = next
      }
      else {
        val tweet = current.get(6) + next.toSeq.filterNot(_ == null).mkString("")
        val row = List(current.get(0), current.get(1), current.get(2), current.get(3),
          current.get(4), current.get(5), tweet)
        current = Row.fromSeq(row)
      }
    }

    val rdd: RDD[Row] = spark.sparkContext.parallelize(list)
    df = spark.createDataFrame(rdd, df.schema)

    //replace URLs and Mentions with general tags
    val urlPattern = "[\"]?http[s]?[^\\s | \" | ;]*"
    val mentionPattern = "@[^\\s|]*"

    df = df.withColumn("tweet", regexp_replace(df("tweet"), urlPattern, "URL_TAG"))
    df = df.withColumn("tweet", regexp_replace(df("tweet"), mentionPattern, "MENTION_TAG"))
    df.select("tweet").collect().take(130).foreach(println(_))

    /*
    //tokenize tweets
    val tokenizer = new RegexTokenizer().setInputCol("tweet").setOutputCol("tokens")
      .setPattern("\\W")
    df = tokenizer.transform(df)

    //remove stopwords
    val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered")
    df = remover.transform(df) //TODO: regex anpassen: hashtags und smilies sollen komplett bleiben

    df.select("filtered").collect().take(100).foreach(println(_))

    //TODO: save to csv file, rename class
*/
  }
}