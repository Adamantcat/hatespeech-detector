/**
  * Created by Julia on 07.06.2017.
  */

import java.util
import java.io._

import scala.collection.JavaConversions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.log4j.Logger
import org.apache.log4j.Level

object LogRegClassifier {

  //method ro convert numerical fields of sequence[String] to Integer
  /*
  def toInt(seq: Seq[String]): Seq[Any] = {
       seq.map(s => s match {
      case _ if s.trim.matches("[0-9]+") => s.toInt
      case _ => s
    })
  }
  */

  def toInt(seq: Seq[String]): Seq[Any] = {
    val tail = seq.last
    val intSeq = seq.slice(0, 6).map(s => s.trim.toInt)
    return intSeq :+ tail
  }

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val data = "/home/kratzbaum/Dokumente/labeled_data.csv"

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()

    val rdd = spark.sparkContext.textFile(data).filter(!_.isEmpty).map(s => s.toLowerCase)
      .map(s => s.split(",", 7).toSeq)

    //restore truncated lines
    val iter = rdd.toLocalIterator
    var current = iter.next()
    val list: util.List[Seq[String]] = new util.ArrayList[Seq[String]]()

    while (iter.hasNext) {
      val next = iter.next()
      if (next.size != 7) {
        val tweet = current.get(6) + " " + next.filterNot(_ == null).mkString("")
        val row = List(current.get(0), current.get(1), current.get(2), current.get(3),
          current.get(4), current.get(5), tweet)
        current = row
      }
      else {
        list.add(current)
        current = next
      }
    }

    val rows = list.tail.map(s => Row.fromSeq(toInt(s)))
    val restoredRDD = spark.sparkContext.parallelize(rows)


    //schema for the dataframe
    val schema = new StructType(Array(StructField("id", IntegerType, nullable = false)
      , StructField("count", IntegerType, nullable = false)
      , StructField("hate_speech", IntegerType, nullable = false)
      , StructField("offensive_language", IntegerType, nullable = false)
      , StructField("neither", IntegerType, nullable = false)
      , StructField("class", IntegerType, nullable = false)
      , StructField("tweet", StringType, nullable = false)
    ))

    var df = spark.createDataFrame(restoredRDD, schema)


    //replace URLs and Mentions with general tags
    val urlPattern = "[\"]?http[s]?[^\\s | \" | ;]*"
    val mentionPattern = "@[^\\s|]*"

    df = df.withColumn("tweet", regexp_replace(df("tweet"), urlPattern, "URL_TAG"))
    df = df.withColumn("tweet", regexp_replace(df("tweet"), mentionPattern, "MENTION_TAG"))

    //tokenize tweets, split at nonword character except & and #
    val tokenizer = new RegexTokenizer().setInputCol("tweet").setOutputCol("tokens")
      .setPattern("[^\\w&#]").setToLowercase(false)
    df = tokenizer.transform(df)

    //remove stopwords
    val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered")
    val stopwords = remover.getStopWords ++ Array[String]("rt", "ff", "#ff")
    remover.setStopWords(stopwords)
    df = remover.transform(df)


    //df.select("id","tweet").collect().take(200).foreach(println(_))

    //TODO: save to file, rename class
    df.write.mode("overwrite").format("json").save("/home/kratzbaum/Dokumente/clean_data")

   val test = spark.read.json("/home/kratzbaum/Dokumente/clean_data")
   test.show

  }
}