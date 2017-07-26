/**
  * Created by Julia on 07.06.2017.
  */

import java.util._

import scala.collection.JavaConversions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.simple._
import org.apache.hadoop.hdfs.util.Diff.ListType
import org.apache.log4j.Logger
import org.apache.log4j.Level

object LogRegClassifier {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("LogReg").getOrCreate()

    import spark.implicits._

    var df = spark.read.json("C:\\Users\\Julia\\Documents\\BA-Thesis\\clean_data")
      .sort("id")

    val getLemmas = udf((tokens: Seq[String]) => {
      new Sentence(tokens).lemmas().toIndexedSeq
    })

    val getPOS = udf((tokens: Seq[String]) => {
      new Sentence(tokens).posTags().toIndexedSeq
    })

    df = df.withColumn("lemmas", getLemmas(df("filtered")))
    df = df.withColumn("pos", getPOS(df("filtered")))

    val unigram = new NGram().setN(1).setInputCol("lemmas").setOutputCol("unigrams")
    df = unigram.transform(df)

    val bigram = new NGram().setN(2).setInputCol("lemmas").setOutputCol("bigrams")
    df = bigram.transform(df)

    val trigram = new NGram().setN(3).setInputCol("lemmas").setOutputCol("trigrams")
    df = trigram.transform(df)

    // tf vectors for uni-, bi-, and trigrams
    val hashingTF_unigram = new HashingTF()
      .setInputCol("unigrams").setOutputCol("rawFeatures_unigrams")

    df = hashingTF_unigram.transform(df)

    val hashingTF_bigram = new HashingTF()
      .setInputCol("bigrams").setOutputCol("rawFeatures_bigrams")

    df = hashingTF_bigram.transform(df)

    val hashingTF_trigram = new HashingTF()
      .setInputCol("trigrams").setOutputCol("rawFeatures_trigrams")

    df = hashingTF_trigram.transform(df)

    //idf vectors for all n-grams
    val idf_unigram = new IDF().setInputCol("rawFeatures_unigrams").setOutputCol("features_unigrams")
    val idfModel_unigram = idf_unigram.fit(df)

    df = idfModel_unigram.transform(df)

    val idf_bigram = new IDF().setInputCol("rawFeatures_bigrams").setOutputCol("features_bigrams")
    val idfModel_bigram = idf_bigram.fit(df)

    df = idfModel_bigram.transform(df)

    val idf_trigram = new IDF().setInputCol("rawFeatures_trigrams").setOutputCol("features_trigrams")
    val idfModel_trigram = idf_trigram.fit(df)

    df = idfModel_trigram.transform(df)

    //define the feature columns to put in the feature vector**
    // val featureCols = Array("features_unigrams", "features_bigrams", "features_trigrams")
    val featureCols = Array("features_trigrams")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    df = assembler.transform(df)

    df = df.withColumnRenamed("class", "label")
    val Array(training, test) = df.randomSplit(Array(0.7, 0.3))

    println("start training")

    val lr = new LogisticRegression().setFeaturesCol("features")
      .setMaxIter(20).setTol(1E-4)
    // Fit the model
    val lrModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficientMatrix} Intercept: ${lrModel.interceptVector}")

    val predictions = lrModel.transform(test)
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("accuracy: " + accuracy)

    evaluator.setMetricName("weightedPrecision")
    val precision = evaluator.evaluate(predictions)
    println("precision: " + precision)

    evaluator.setMetricName("weightedRecall")
    val recall = evaluator.evaluate(predictions)
    println("recall: " + recall)

    evaluator.setMetricName("f1")
    val f1 = evaluator.evaluate(predictions)
    println("f1-score: " + f1)


    /* val paramGrid = new ParamGridBuilder().build() // No parameter search
     val cv = new CrossValidator()
       .setEstimator(lr)
       .setEvaluator(new MulticlassClassificationEvaluator())
       .setEstimatorParamMaps(paramGrid)
       .setNumFolds(5)

     val cvModel = cv.fit(df)
     cvModel.avgMetrics.foreach(println(_))
     */
  }
}