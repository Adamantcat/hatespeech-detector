/**
  * Created by kratzbaum on 01.06.17.
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD


object Test {

  def main(args : Array[String]): Unit = {
    println("Hello World!")
    println("Hello Scala")

    // Create a local Spark config for testing, setting the number of cores explicitly.
    val sparkConfig = new SparkConf().setMaster("local").setAppName("Test")
    // Initialize the Spark context
    val sc = new SparkContext(sparkConfig)

    val data = Array(1, 2, 3, 4, 5)
    val distData = sc.parallelize(data)

    val res = distData.reduce(_+_)
    println(res)
  }

}
