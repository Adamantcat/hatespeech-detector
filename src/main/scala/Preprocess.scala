import java.util

import scala.collection.JavaConversions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.log4j.Logger
import org.apache.log4j.Level
/**
  * Created by kratzbaum on 04.07.17.
  */
object Preprocess {

  //convert first 6 fields of Sequence to integer
  def toInt(seq: Seq[String]): Seq[Any] = {
    val last = seq.last
    val intSeq = seq.slice(0, 6).map(s => s.trim.toInt)
    intSeq :+ last
  }

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val data = "/home/kratzbaum/Dokumente/labeled_data.csv"

    val spark = SparkSession.builder.master("local[*]")
      .appName("Preprocess").getOrCreate()

    val rdd = spark.sparkContext.textFile(data).filter(!_.isEmpty).map(s => s.toLowerCase)
      .map(s => s.split(",", 7).toSeq)

    //restore truncated lines
    val iter = rdd.toLocalIterator
    var current = iter.next()
    var next = current
    val list: util.List[Seq[String]] = new util.ArrayList[Seq[String]]()

    while (iter.hasNext) {
      next = iter.next()
      if (next.size != 7) {
        val tweet = current.get(6) + " " + next.filterNot(_ == null).mkString("")
        val row = current.init :+ tweet
        current = row
      }
      else {
        list.add(current)
        current = next
      }
    }
    list.add(next) //add last element

    val rows = list.tail.map(s => Row.fromSeq(toInt(s))) //convert to list[Row], drop header
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

    //save as file
    df.write.mode("overwrite").format("json").save("/home/kratzbaum/Dokumente/clean_data")
    val test = spark.sqlContext.read.json("/home/kratzbaum/Dokumente/clean_data")
    test.sort("id").show
  }
}
