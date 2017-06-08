/**
  * Created by Julia on 07.06.2017.
  */

import java.util

import scala.collection.JavaConversions._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, Row, SQLContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.log4j.Logger
import org.apache.log4j.Level


object LogRegClassifier {

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val data = "C:\\Users\\Julia\\Documents\\BA-Thesis\\labeled_data.csv"

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()

    val df = spark.read.option("header", "true").csv(data)
   // val filtered = df.filter(_.size > 0) //remove empty lines

    val iter = df.toLocalIterator()
    var current: Row = iter.next() // initialize with first element
    var list: util.List[Row] = new util.ArrayList[Row]()

     while(iter.hasNext) {
     val next = iter.next()
      if(!next.anyNull) {
        list.add(current)
        current = next
      }
      else {
        val tweet = current.get(6) + next.toString()
        val row = List(current.get(0), current.get(1), current.get(2), current.get(3),
        current.get(4), current.get(5), tweet)
        current = Row.fromSeq(row)
      }
    }

    val rdd: RDD[Row] = spark.sparkContext.parallelize(list)
    val cleanDF = spark.createDataFrame(rdd, df.schema)
    //rdd.collect().take(100).foreach(println(_))

   cleanDF.show(100)

  }
}