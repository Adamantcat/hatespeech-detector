/**
AUTHOR: Julia Koch
PURPOSE: Trains a logistic regression classifier
License: Copyright [yyyy] Julia Koch

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
  */

import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, MultiMetricCrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LogRegClassifier {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkSession.builder.master("local[*]")
      .appName("LogReg").getOrCreate()

    var df = spark.read.json("/home/mau/Documents/clean_data")
      .sort("id")

    val Array(train, test) = df.cache.randomSplit(Array(0.8, 0.2))

    val trainCounts = train.select("label").collect.groupBy(identity).map(r => (r._1.getLong(0), r._2.length))
    println("class counts in training data:")
    trainCounts.foreach(l => println(l._1 + ": " + l._2))

    println(trainCounts)

    //save test data to file for later use
   test.write.format("json").save("/home/mau/Documents/test")

    //assign weights to underrepresented classes, similar effect to oversampling
    //pretend all classes are equally distributed
    val setWeights = udf((label: Int) => {
      label match {
        case 0 => trainCounts.getOrElse(1, 0).toDouble / trainCounts.getOrElse(0, 0)
        case 1 => 1
        case 2 => trainCounts.getOrElse(1, 0).toDouble / trainCounts.getOrElse(2, 0)
        case _ => 0.0
      }
    })

    val training = train.withColumn("weight", setWeights(train("label")))

    //tf-idf features for bigrams and trigrams
    val hashingTF_bigrams = new HashingTF().setInputCol("bigrams").setOutputCol("bigramTF")
    val idf_bigrams = new IDF().setInputCol("bigramTF").setOutputCol("bigramFeatures")

    val hashingTF_trigrams = new HashingTF().setInputCol("trigrams").setOutputCol("trigramTF")
    val idf_trigrams = new IDF().setInputCol("trigramTF").setOutputCol("trigramFeatures")

    //combine bigram and trigram features to one large vector
    val assembler = new VectorAssembler().setInputCols(Array("bigramFeatures", "trigramFeatures"))
      .setOutputCol("combinedFeatures")

    println("start parameter tuning")

    //set up for training
    val lr = new LogisticRegression().setMaxIter(10).setTol(1E-4).setPredictionCol("prediction")

    val pipeline = new Pipeline()
      .setStages(Array(hashingTF_bigrams, hashingTF_trigrams, idf_bigrams, idf_trigrams, assembler, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.featuresCol, Array("bigramFeatures", "trigramFeatures", "combinedFeatures"))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.3, 0.7, 1.0))
      .addGrid(lr.regParam, Array(0.0, 0.01, 0.001))
      .addGrid(lr.weightCol, Array("weight", ""))
      .build()


    val cv = new MultiMetricCrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1")
        .setLabelCol("label").setPredictionCol("prediction"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = cv.fit(training)
    val results = cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)
    results.foreach(println(_))


    //save results for f1 score to file
    val pw = new PrintWriter(new File("/home/mau/Documents/results_f1.txt"))
    results.foreach(s => pw.write(s.toString + "\n"))
    pw.close

    //save best model
    val best_model = cvModel.bestModel.asInstanceOf[PipelineModel]
    best_model.save("/home/mau/Documents/best_model")
  }
}