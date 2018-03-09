import java.io.File

import org.apache.spark.sql.DataFrame
import utils.SparkUtil.RandomForestExecutor
import utils.{Setup, SparkUtil}

object Program {
  lazy val data1DF: DataFrame = SparkUtil.session.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", ",")
    .csv(Setup.settings.data1Location.get)

  def main(args: Array[String]): Unit = {
    println(s"${Console.MAGENTA}${Console.BOLD}${Setup.settings.sparkAppName.get} started.${Console.RESET}")

    if (!Setup.settings.hasAllDefined) {
      println(s"${Console.RED}ERROR - Missing configuration entry. To start the process, all configuration entries must be defined.${Console.RESET}")
      Setup.settings.printMissingFields()
      System.exit(1)
    }

    System.setProperty("hadoop.home.dir", new File(Setup.settings.hadoopHome.get).getCanonicalPath)
    println(s"${Console.RED}Spark version: ${Console.RESET}${SparkUtil.session.version}")

    SparkUtil.printExecutionTime(runRandomForest())

    SparkUtil.session.close()
  }

  def runRandomForest(): Unit = {
    val targetDF = data1DF.select(SparkUtil.DataColumns.text.toString, SparkUtil.DataColumns.topic.toString)
      .withColumnRenamed(SparkUtil.DataColumns.topic.toString, SparkUtil.DataColumns.label.toString)

    val pikachu = new RandomForestExecutor(targetDF)
      .addRegexTokenizer("[\\W_]+", Predef.int2Integer(4), SparkUtil.DataColumns.text.toString, SparkUtil.DataColumns.tokens.toString)
      .addStopWordsRemover(SparkUtil.DataColumns.tokens.toString, SparkUtil.DataColumns.filtered.toString)
      .removeColumn(SparkUtil.DataColumns.tokens.toString)
      .addLabelIndexer(SparkUtil.DataColumns.label.toString, SparkUtil.DataColumns.indexedLabel.toString, "skip")
      .addCountVectorizer(SparkUtil.DataColumns.filtered.toString, SparkUtil.DataColumns.features.toString, Predef.int2Integer(1000))
      .applyRandomSplit(Array(.8, .2), 1)
      .convertLabelsToString(SparkUtil.DataColumns.prediction.toString, SparkUtil.DataColumns.predictedLabel.toString)
      .addRandomForestClassifier(SparkUtil.DataColumns.indexedLabel.toString, SparkUtil.DataColumns.features.toString)
      .addParameterGrid(Array(2, 5, 8, 12), Array(6, 12))
      .addMulticlassClassificationEvaluator(SparkUtil.DataColumns.indexedLabel.toString, SparkUtil.DataColumns.prediction.toString, "accuracy")
      .addCrossValidator(3)
      .addPipelineStages()
      .createPredictions
      .getModelWithBestHyperConfig
      .getAccuracy

    pikachu.build

    pikachu.predictionsDataFrame.select(SparkUtil.DataColumns.label.toString
      , SparkUtil.DataColumns.predictedLabel.toString).show()

    pikachu.modelWithBestHyperParameterConfig.foreach(println)

    println(s"\rTest Accuracy = ${pikachu.accuracy}")

  }
}
