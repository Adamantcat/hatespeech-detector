/**
  * Created by Julia on 07.06.2017.
  */

import java.util

import scala.util.matching.Regex
import scala.collection.JavaConversions._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext

object LogRegClassifier {

  //method ro convert numerical fields of sequence[String] to Integer
  def toInt(seq: Seq[String]): Seq[Any] = {
    seq.map(s => s match {
      case _ if s.matches("[0-9]+") => s.toInt
      case _ => s
    })
  }

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val data = "C:\\Users\\Julia\\Documents\\BA-Thesis\\labeled_data.csv"

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()

    import spark.implicits._

    //read file, convert to lowercase, split fields, drop header line
    val rdd = spark.sparkContext.textFile(data).filter(!_.isEmpty).map(s => s.toLowerCase)
      .map(s => s.split(",", 7).toSeq).map(s => toInt(s)).map(s => Row.fromSeq(s))
      .mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

    val iter = rdd.toLocalIterator
    var current = iter.next()
    val list: util.List[Row] = new util.ArrayList[Row]()

    while (iter.hasNext) {
      val next = iter.next()
      if (next.size < 7) {
        val tweet = current.get(6) + next.toSeq.filterNot(_ == null).mkString("")
        val row = List(current.get(0), current.get(1), current.get(2), current.get(3),
          current.get(4), current.get(5), tweet)
        current = Row.fromSeq(row)
      }
      else {
        list.add(current)
        current = next
      }
    }
    val restoredRdd = spark.sparkContext.parallelize(list)

    val schema = new StructType(Array(StructField("id", IntegerType, nullable = false)
      , StructField("count", IntegerType, nullable = false)
      , StructField("hate_speech", IntegerType, nullable = false)
      , StructField("offensive_language", IntegerType, nullable = false)
      , StructField("neither", IntegerType, nullable = false)
      , StructField("class", IntegerType, nullable = false)
      , StructField("tweet", StringType, nullable = false)
    ))

    var df = spark.createDataFrame(restoredRdd, schema)
    df.printSchema()
    df.show(50)

    //replace URLs and Mentions with general tags
   val urlPattern = "[\"]?http[s]?[^\\s | \" | ;]*"
    // val mentionPattern = "@[^\\s|]*"

     df = df.withColumn("tweet", regexp_replace(df("tweet"), urlPattern, "URL_TAG"))
     //df = df.withColumn("tweet", regexp_replace(df("tweet"), mentionPattern, "MENTION_TAG"))
     df.select("tweet").collect().take(130).foreach(println(_))
     // df.printSchema()


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