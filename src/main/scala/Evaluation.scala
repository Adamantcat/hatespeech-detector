/**
  * Created by kratzbaum on 08.08.17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

object Evaluation {

  def main(args: Array[String]): Unit = {

    //reduce verbosity of logger
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("Evaluate").getOrCreate()

    var test = spark.read.json("/home/kratzbaum/Dokumente/test")
      .sort("id")


    val idf_trigram = new IDF().setInputCol("new_rawFeatures_trigrams").setOutputCol("new_features_trigrams")
    val idfModel_trigram = idf_trigram.fit(test)

    test = idfModel_trigram.transform(test)


    val assembler = new VectorAssembler().setInputCols(Array("new_features_trigrams")).setOutputCol("features")
    test = assembler.transform(test)

    val best_model = LogisticRegressionModel.load("/home/kratzbaum/Dokumente/best_model")
    val predictions = best_model.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
    val f1 = evaluator.evaluate(predictions)
    println("f1 score: " + f1)
  }

}
