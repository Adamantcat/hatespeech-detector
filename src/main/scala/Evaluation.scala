/**
  * Created by kratzbaum on 08.08.17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

import scala.collection.Map

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

    //word statistics for correctly and incorrectly classified tweets

    //remove stopwords, not used by classifier
    val remover = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filtered")
    val stopwords = remover.getStopWords ++ Array[String](".", ",", ";", ":", "?", "!", "\"", "\'")
    remover.setStopWords(stopwords)
    val stats = remover.transform(predictions)


    val correctHate = stats.filter("label == 0 AND prediction == 0.0")
    val correctOffensive = stats.filter("label == 1 AND prediction == 1.0")
    val correctNeither = stats.filter("label == 2 AND prediction == 2.0")

    val trueHate_predOffensive = stats.filter("label == 0 AND prediction == 1.0")
    val trueHate_predNeither = stats.filter("label == 0 AND prediction == 2.0")

    val trueOffensive_predHate = stats.filter("label == 1 AND prediction == 0.0")
    val trueOffensive_predNeither = stats.filter("label == 1 AND prediction == 2.0")

    val trueNeither_predHate = stats.filter("label == 2 AND prediction == 0.0")
    val trueNeither_predOffensive = stats.filter("label == 2 AND prediction == 1.0")



    val correctHateFreq = correctHate.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n Correctly classified hate words:")
    correctHateFreq.take(20).foreach(println(_))


    val correctOffensiveFreq = correctOffensive.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n Correctly classified offensive words:")
    correctOffensiveFreq.take(20).foreach(println(_))


    val correctNeitherFreq = correctNeither.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n Correctly classified inoffensive words:")
    correctNeitherFreq.take(20).foreach(println(_))


    val trueHate_predOffFreq = trueHate_predOffensive.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: hate, but predicted offensive:")
    trueHate_predOffFreq.take(20).foreach(println(_))


    val trueHate_predNeitherFreq = trueHate_predNeither.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: hate, but predicted neither:")
    trueHate_predNeitherFreq.take(20).foreach(println(_))


    val trueOff_predHateFreq = trueOffensive_predHate.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: offensive, but predicted hate:")
    trueOff_predHateFreq.take(20).foreach(println(_))


    val trueOff_predNeitherFreq = trueOffensive_predNeither.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: offensive, but predicted neither:")
    trueOff_predNeitherFreq.take(20).foreach(println(_))


    val trueNeither_predHateFreq = trueNeither_predHate.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: neither, but predicted hate:")
    trueNeither_predHateFreq.take(20).foreach(println(_))

    val trueNeither_predOffFreq = trueNeither_predOffensive.select("filtered").rdd.flatMap(x => x.getSeq[String](0)).map(word => (word, 1))
      .reduceByKey(_+_).sortBy(_._2, numPartitions = 1, ascending = false)

    println("\n true cat: neither, but predicted offensive:")
    trueNeither_predOffFreq.take(20).foreach(println(_))
  }
}
