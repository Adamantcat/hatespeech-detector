/**
AUTHOR: Julia Koch
PURPOSE: This modification of Spark CrossValidator is a workaround to print results
for precision and recall during cross validation and save result to file
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
package org.apache.spark.ml.tuning

import java.io.PrintWriter
import java.io.File
import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.ml._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Dataset}


class MultiMetricCrossValidator extends org.apache.spark.ml.tuning.CrossValidator{

  private val f2jBLAS = new F2jBLAS

  // @Since("2.0.0")
  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)

    //modification: additional evluators
    val eval_p = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
    val eval_r = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
    val precisions = new Array[Double](epm.length)
    val recalls = new Array[Double](epm.length)
    //mod end

    val instr = Instrumentation.create(this, dataset)
    instr.logParams(numFolds, seed)
    logTuningParams(instr)

    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
        logWarning(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric

        //modification: evaluate precision and recall during cross validation
        val precision = eval_p.evaluate(models(i).transform(validationDataset, epm(i)))
        logWarning(s"Got precision $precision for model trained with ${epm(i)}.")
        val recall = eval_r.evaluate(models(i).transform(validationDataset, epm(i)))
        logWarning(s"Got recall $recall for model trained with ${epm(i)}.")

        precisions(i) += precision
        recalls(i) += recall

        //mod end
        i += 1
      }
      validationDataset.unpersist()
    }
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    logWarning(s"Average cross-validation metrics: ${metrics.toSeq}")

    //modification: calculate average results for precision and recall and save them to files
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), precisions, 1)
    logWarning(s"Average cross-validation precision: ${precisions.toSeq}")

    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), recalls, 1)
    logWarning(s"Average cross-validation recall: ${recalls.toSeq}")

    val results_p = epm.zip(precisions)
    val results_r = epm.zip(recalls)

    val pw_p = new PrintWriter(new File("/home/mau/Documents/results_precision.txt" ))
    results_p.foreach(s => pw_p.write(s.toString + "\n"))
    pw_p.close

    val pw_r = new PrintWriter(new File("/home/mau/Documents/results_recalls.txt" ))
    results_r.foreach(s => pw_r.write(s.toString + "\n"))
    pw_r.close

    //mod end
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    instr.logSuccess(bestModel)
    copyValues(new CrossValidatorModel(uid, bestModel, metrics).setParent(this))
  }

}

