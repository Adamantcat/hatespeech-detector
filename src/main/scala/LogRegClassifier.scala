/**
  * Created by Julia on 07.06.2017.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

object LogRegClassifier {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("LogReg").getOrCreate()

    var df = spark.read.json("/home/kratzbaum/Dokumente/clean_data")
      .sort("id")

    df = df.withColumnRenamed("class", "label")
    val Array(training, test) = df.cache.randomSplit(Array(0.8, 0.2))


    //*******test
    /*
        val numIds = df.select("id").collect
        val setIds = numIds.toSet

        println("duplicate ids:" + (numIds.length - setIds.size))

        training.sort("id").show()
        test.sort("id").show()

        val testlist = test.collectAsList()

       val overlap = training.filter(row => testlist.contains(row)).sort("id")
        println("overlap: " + overlap.count())
        println("test: " + test.count)

        println(overlap)
        overlap.select("id").sort("id").foreach(println(_))

        val ol = training.intersect(test).collect()
        println(ol.length)
    */

    //***********

    // tf vectors for uni-, bi-, and trigrams
    /*
     val hashingTF_unigram = new HashingTF()
       .setInputCol("unigrams").setOutputCol("rawFeatures_unigrams")

     df = hashingTF_unigram.transform(df)

     val hashingTF_bigram = new HashingTF()
       .setInputCol("bigrams").setOutputCol("rawFeatures_bigrams")

     df = hashingTF_bigram.transform(df)
     */

    val hashingTF_trigram = new HashingTF()
      .setInputCol("trigrams").setOutputCol("rawFeatures_trigrams")


    //idf vectors for all n-grams

    /*
    val idf_unigram = new IDF().setInputCol("rawFeatures_unigrams").setOutputCol("features_unigrams")
    val idfModel_unigram = idf_unigram.fit(df)

    df = idfModel_unigram.transform(df)

    val idf_bigram = new IDF().setInputCol("rawFeatures_bigrams").setOutputCol("features_bigrams")
    val idfModel_bigram = idf_bigram.fit(df)

    df = idfModel_bigram.transform(df)
    */

    val idf_trigram = new IDF().setInputCol(hashingTF_trigram.getOutputCol).setOutputCol("features_trigrams")

    //define the feature columns to put in the feature vector**
    // val featureCols = Array("features_unigrams", "features_bigrams", "features_trigrams")
    val featureCols = Array(idf_trigram.getOutputCol)
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")


    //save test data to file for later use
    test.write.mode("overwrite").format("json").save("/home/kratzbaum/Dokumente/test")

    println("start parameter tuning")

    val lr = new LogisticRegression().setFeaturesCol("features")
      .setMaxIter(5).setTol(1E-4).setPredictionCol("prediction")

    val pipeline = new Pipeline()
      .setStages(Array(hashingTF_trigram, idf_trigram, assembler, lr))

    val paramGrid = new ParamGridBuilder().build()

    /*
     val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0, 0.3, 0.7, 1.0))
      .addGrid(lr.regParam, Array(0.0, 0.01, 0.001))
      .build()
      */

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1")
        .setLabelCol("label").setPredictionCol("prediction"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(training)
    val avgMetrics = cvModel.avgMetrics

    avgMetrics.foreach(println(_))

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val bestParams = bestModel.explainParams()

    println("best Parameters: ")
    for(i <- 0 until bestModel.stages.length) {
      println(bestModel.stages(i).explainParams() + "\n")
    }

    bestModel.save("/home/kratzbaum/Dokumente/best_model")


    val predictions = bestModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
      .setLabelCol("label").setPredictionCol("prediction")
    val f1 = evaluator.evaluate(predictions)
    println("f1 score: " + f1)

    evaluator.setMetricName("weightedPrecision")
    val precision = evaluator.evaluate(predictions)
    println("weightedPrecision: " + precision)

    evaluator.setMetricName("weightedRecall")
    val recall = evaluator.evaluate(predictions)
    println("weightedPrecision: " + recall)

    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy: " + accuracy)

  }
}