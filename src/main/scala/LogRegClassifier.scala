/**
  * Created by Julia on 07.06.2017.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
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
    val Array(training, test) = df.randomSplit(Array(0.8, 0.2))


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
      .setMaxIter(10).setTol(1E-4)

    val pipeline = new Pipeline()
      .setStages(Array(hashingTF_trigram, idf_trigram, assembler, lr))

    val paramGrid = new ParamGridBuilder().build()

    /* val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0, 0.3, 0.7, 1.0))
      .addGrid(lr.regParam, Array(0.1, 0.01, 0.001))
      .build()
      */

    /*val model = pipeline.fit(training)

    val predictions = model.transform(test)
    val eval = new MulticlassClassificationEvaluator().setMetricName("f1")
    val f1 = eval.evaluate(predictions)
    println("f1: " + f1)*/



     val cv = new CrossValidator()
       .setEstimator(pipeline)
       .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1"))
       .setEstimatorParamMaps(paramGrid)
       .setNumFolds(5)

    val cvModel = cv.fit(training)
    val avgMetrics = cvModel.avgMetrics

    avgMetrics.foreach(println(_))

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val bestParams = bestModel.explainParams()

    println("best Parameters: " + bestParams)
    bestModel.save("/home/kratzbaum/Dokumente/best_model")


    val predictions = bestModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
    val f1 = evaluator.evaluate(predictions)
    println("f1 score: " + f1)

  }
}