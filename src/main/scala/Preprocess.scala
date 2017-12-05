import java.util

import edu.stanford.nlp.simple._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{NGram, StopWordsRemover}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.JavaConversions._

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
      , StructField("label", IntegerType, nullable = false)
      , StructField("tweet", StringType, nullable = false)
    ))

    var df = spark.createDataFrame(restoredRDD, schema)

    //replace URLs and Mentions with general tags
    val urlPattern = "[\"]?http[s]?[^\\s | \" | ;]*"
    val mentionPattern = "@[^\\s|]*"

    df = df.withColumn("tweet", regexp_replace(df("tweet"), urlPattern, "url_tag"))
    df = df.withColumn("tweet", regexp_replace(df("tweet"), mentionPattern, "mention_tag"))

    //tokenization, preserve punctuation, matches:
    //emoticons starting with &#, hashtags and escape characters (e.g. &#1234, #halloween, &amp)
    //sequences of word characters interrupted by * (e.g. n*gga)
    //sequences of word characters (normal words)
    //non-word and non-whitespace characters (puntuation)
    val regexp = "[&#]*\\w+[\\*]*\\w+|\\w+|[^\\w\\s]".r
    val getTokens = udf((tweet: String) => {
      regexp.findAllIn(tweet).toSeq
    })

    df = df.withColumn("tokens", getTokens(df("tweet")))

    val getLemmas = udf((tokens: Seq[String]) => {
     new Sentence(tokens).lemmas().toIndexedSeq
    })

   /* val getPOS = udf((tokens: Seq[String]) => {
      new Sentence(tokens).posTags().toIndexedSeq
    }) */

    df = df.withColumn("lemmas", getLemmas(df("tokens")))
    //df = df.withColumn("pos", getPOS(df("tokens")))

    val unigram = new NGram().setN(1).setInputCol("lemmas").setOutputCol("unigrams")
    df = unigram.transform(df)

    val bigram = new NGram().setN(2).setInputCol("lemmas").setOutputCol("bigrams")
    df = bigram.transform(df)

    val trigram = new NGram().setN(3).setInputCol("lemmas").setOutputCol("trigrams")
    df = trigram.transform(df)

    //save as file
    df.write.mode("overwrite").format("json").save("/home/kratzbaum/Dokumente/clean_data")


    //******************************************************************************
    //some document statistics

    //total number of tweets
    println("total num tweets: " + df.count())

    //class counts
    val totalCounts = df.select("label").collect.groupBy(identity).mapValues(_.size)
    println("class counts in complete dataset:")
    totalCounts.foreach(l => println(l._1 + ": " + l._2))


    //remove stopwords, not used by classifier
    val remover = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filtered")
    val stopwords = remover.getStopWords ++ Array[String](".", ",", ";", ":", "?", "!", "\"", "\'")
    remover.setStopWords(stopwords)
    df = remover.transform(df)

    //compute most frequent words for each class
    val hate = df.filter("label == 0")
    val offensive = df.filter("label == 1")
    val neither = df.filter("label == 2")

    val hateFreq = hate.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
    .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("Top 20 hate words:")
    hateFreq.take(20).foreach(println(_))

    val offenseFreq = offensive.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("Top 20 offensive words")
    offenseFreq.take(20).foreach(println(_))

    val neutralFreq = neither.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("Top 20 neutral words")
    neutralFreq.take(20).foreach(println(_))
  }
}
