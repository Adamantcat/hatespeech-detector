/**
  * Created by kratzbaum on 08.08.17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
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

    import spark.implicits._

    val best_model = PipelineModel.load("/home/kratzbaum/Dokumente/best_model")

    println("best Parameters: ")
    for (i <- 0 until best_model.stages.length) {
      println(best_model.stages(i).explainParams() + "\n")
    }

    val predictions = best_model.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
      .setLabelCol("label").setPredictionCol("prediction")
    val f1 = evaluator.evaluate(predictions)
    println("f1 score: " + f1)

    evaluator.setMetricName("weightedPrecision")
    val precision = evaluator.evaluate(predictions)
    println("weighted Precision: " + precision)

    evaluator.setMetricName("weightedRecall")
    val recall = evaluator.evaluate(predictions)
    println("weighted Recall: " + recall)

    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy: " + accuracy)

    //rdd based mllib has some additional useful metrics
    //to use these, dataframe needs to be converted to rdd
    val predictionsAndLabels = predictions.select("prediction", "label")
      .map(r => new Tuple2[Double, Double](r.getDouble(0), r.getLong(1).toDouble)).rdd

    val metrics = new MulticlassMetrics(predictionsAndLabels)
    val falsePositives = metrics.weightedFalsePositiveRate
    println("Weighted false positive rate: " + falsePositives + "\n")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }
    println()

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }
    println()

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }
    println()

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }
    println()

    //basic counts
    val labelCounts = predictionsAndLabels.values.groupBy(identity).mapValues(_.size)
    println("total entries in test set: " + test.count())
    println("counts per label")
    labelCounts.foreach(l => println(l._1 + ": " + l._2))
    println()

    //prints labels in ascending order: 0.0 1.0 2.0
    //                              0.0
    //                              1.0
    //                              2.0
    val confusionMatrix = metrics.confusionMatrix
    println(confusionMatrix)
    println()

  }
}
