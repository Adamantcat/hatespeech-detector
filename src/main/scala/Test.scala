/**
  * Created by kratzbaum on 01.06.17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object Test {

  def main(args : Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()

    val test = spark.read.json("C:\\Users\\Julia\\Documents\\BA-Thesis\\clean_data")
    test.show
    println(test.count())
  }

}
