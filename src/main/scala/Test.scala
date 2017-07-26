/**
  * Created by kratzbaum on 01.06.17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row


object Test {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("Example").getOrCreate()

    // Prepare training data from a list of (id, text, label) tuples.
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 2.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 2.0),
      (7L, "was mapreduce", 2.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val evaluator = new MulticlassClassificationEvaluator
    evaluator.setMetricName("accuracy")

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training)

    cvModel.avgMetrics.foreach(println(_))

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    // Explain params for each stage
    val bestHashingTFNumFeatures = bestModel.stages(1).asInstanceOf[HashingTF].explainParams
    val bestLrRegParamFeatures = bestModel.stages(2).asInstanceOf[LogisticRegressionModel].explainParams

    println("best hasshingTF: " + bestHashingTFNumFeatures)
    println("best lr: " + bestLrRegParamFeatures)

  /*  val predictions = bestModel.transform(training)
    predictions.show()
   val f1 = evaluator.evaluate(predictions)
    println("f1: " + f1)
    */

     // Prepare test documents, which are unlabeled (id, text) tuples.
     val test = spark.createDataFrame(Seq(
       (4L, "spark i j k"),
       (5L, "l m n"),
       (6L, "mapreduce spark"),
       (7L, "apache hadoop")
     )).toDF("id", "text")

    import spark.implicits._

    val predictions = bestModel.transform(training)
    bestModel.save("C:\\Users\\Julia\\Documents\\BA-Thesis\\best_model")
    predictions.printSchema()

    val predictionAndLabels = predictions.map(r => new Tuple2[Double, Double](r.getDouble(2), r.getDouble(7))).rdd

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }
}
