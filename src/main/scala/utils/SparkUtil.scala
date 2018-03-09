package utils

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StopWordsRemover, _}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object SparkUtil {
  lazy val session: SparkSession = SparkSession
    .builder()
    .appName(Setup.settings.sparkAppName.get)
    .master(Setup.settings.sparkMaster.get)
    .enableHiveSupport()
    .getOrCreate()

  def printExecutionTime[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val milliseconds = (t1 - t0) / 1000000
    println(s"\r${Console.BOLD}Execution duration: ${(milliseconds / (1000 * 60 * 60)) % 24}:${(milliseconds / (1000 * 60)) % 60}:${(milliseconds / 1000) % 60}${Console.RESET}")
    result
  }

  object DataColumns extends Enumeration {
    type dataIndices = Value
    val id: SparkUtil.DataColumns.Value = Value(0)
    val text: SparkUtil.DataColumns.Value = Value(1)
    val topic: SparkUtil.DataColumns.Value = Value(2)
    val label: SparkUtil.DataColumns.Value = Value(3)
    val tokens: SparkUtil.DataColumns.Value = Value(4)
    val filtered: SparkUtil.DataColumns.Value = Value(5)
    val indexedLabel: SparkUtil.DataColumns.Value = Value(7)
    val features: SparkUtil.DataColumns.Value = Value(8)
    val prediction: SparkUtil.DataColumns.Value = Value(9)
    val predictedLabel: SparkUtil.DataColumns.Value = Value(10)
  }

  abstract class RandomForest

  abstract class RandomForestModelBuilder {
    var activeDataFrame: DataFrame
    var trainDataFrame: DataFrame
    var testDataFrame: DataFrame
    var predictionsDataFrame: DataFrame
    var tokenizedDataFrame: DataFrame
    var filteredDataFrame: DataFrame
    var finalDataFrame: DataFrame
    var regexTokenizer: RegexTokenizer
    var remover: StopWordsRemover
    var labelIndexer: StringIndexerModel
    var vectorizer: CountVectorizerModel
    var splitter: Array[Dataset[Row]]
    var labelConverter: IndexToString
    var randomForestClassifier: RandomForestClassifier
    var parameterGrid: Array[ParamMap]
    var multiclassClassificationEvaluator: MulticlassClassificationEvaluator
    var crossValidator: CrossValidator
    var pipeline: Pipeline
    var accuracy: Double
    var modelWithBestHyperParameterConfig: Array[(ParamMap, Double)]
    var pipelineModel: PipelineModel

    def addRegexTokenizer(pattern: String, minTokenLength: Integer, inputColumn: String, outputColumn: String): RandomForestModelBuilder

    def addStopWordsRemover(inputColumn: String, outputColumn: String): RandomForestModelBuilder

    def removeColumn(columnName: String): RandomForestModelBuilder

    def addLabelIndexer(inputColumn: String, outputColumn: String, handleInvalid: String): RandomForestModelBuilder

    def addCountVectorizer(inputColumn: String, outputColumn: String, vocabularySize: Integer): RandomForestModelBuilder

    def applyRandomSplit(weights: Array[Double], seed: Long): RandomForestModelBuilder

    def convertLabelsToString(inputColumn: String, outputColumn: String): RandomForestModelBuilder

    def addRandomForestClassifier(labelColumn: String, featuresColumn: String): RandomForestModelBuilder

    def addParameterGrid(maxDepth: Array[Int], numberOfTrees: Array[Int]): RandomForestModelBuilder

    def addMulticlassClassificationEvaluator(labelColumn: String, predictionColumn: String, metricName: String): RandomForestModelBuilder

    def addCrossValidator(numberOfFolds: Integer): RandomForestModelBuilder

    def addPipelineStages(): RandomForestModelBuilder

    def createPredictions: RandomForestModelBuilder

    def getAccuracy: RandomForestModelBuilder

    def getModelWithBestHyperConfig: RandomForestModelBuilder

    def build: RandomForest
  }

  class RandomForestPrediction(builder: RandomForestModelBuilder) extends RandomForest {
    var activeDataFrame: DataFrame = builder.activeDataFrame
    var trainDataFrame: DataFrame = builder.trainDataFrame
    var testDataFrame: DataFrame = builder.testDataFrame
    var predictionsDataFrame: DataFrame = builder.predictionsDataFrame
    var tokenizedDataFrame: DataFrame = builder.tokenizedDataFrame
    var filteredDataFrame: DataFrame = builder.filteredDataFrame
    var finalDataFrame: DataFrame = builder.finalDataFrame
    val regexTokenizer: RegexTokenizer = builder.regexTokenizer
    val remover: StopWordsRemover = builder.remover
    val labelIndexer: StringIndexerModel = builder.labelIndexer
    val vectorizer: CountVectorizerModel = builder.vectorizer
    val splitter: Array[Dataset[Row]] = builder.splitter
    val labelConverter: IndexToString = builder.labelConverter
    val randomForestClassifier: RandomForestClassifier = builder.randomForestClassifier
    val parameterGrid: Array[ParamMap] = builder.parameterGrid
    val multiclassClassificationEvaluator: MulticlassClassificationEvaluator = builder.multiclassClassificationEvaluator
    val crossValidator: CrossValidator = builder.crossValidator
    val pipeline: Pipeline = builder.pipeline
    val accuracy: Double = builder.accuracy
    var modelWithBestHyperParameterConfig: Array[(ParamMap, Double)] = builder.modelWithBestHyperParameterConfig
    var pipelineModel: PipelineModel = builder.pipelineModel

  }

  class RandomForestExecutor(targetDataFrame: DataFrame) extends RandomForestModelBuilder {
    override var activeDataFrame: DataFrame = targetDataFrame
    override var trainDataFrame: DataFrame = _
    override var testDataFrame: DataFrame = _
    override var predictionsDataFrame: DataFrame = _
    override var tokenizedDataFrame: DataFrame = _
    override var filteredDataFrame: DataFrame = _
    override var finalDataFrame: DataFrame = _
    override var regexTokenizer: RegexTokenizer = _
    override var remover: StopWordsRemover = _
    override var labelIndexer: StringIndexerModel = _
    override var vectorizer: CountVectorizerModel = _
    override var splitter: Array[Dataset[Row]] = _
    override var labelConverter: IndexToString = _
    override var randomForestClassifier: RandomForestClassifier = _
    override var parameterGrid: Array[ParamMap] = _
    override var multiclassClassificationEvaluator: MulticlassClassificationEvaluator = _
    override var crossValidator: CrossValidator = _
    override var pipeline: Pipeline = _
    override var accuracy: Double = _
    override var modelWithBestHyperParameterConfig: Array[(ParamMap, Double)] = _
    override var pipelineModel: PipelineModel = _

    //.addRegexTokenizer("[\\W_]+", Predef.int2Integer(4), SparkUtil.DataColumns.text.toString, SparkUtil.DataColumns.tokens.toString)
    override def addRegexTokenizer(pattern: String, minTokenLength: Integer, inputColumn: String, outputColumn: String): RandomForestModelBuilder = {
      this.regexTokenizer = new RegexTokenizer()
        .setPattern(pattern)
        .setMinTokenLength(Predef.Integer2int(minTokenLength))
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)

      this.tokenizedDataFrame = this.regexTokenizer.transform(activeDataFrame)

      this
    }

    //.addStopWordsRemover(SparkUtil.DataColumns.tokens.toString, SparkUtil.DataColumns.filtered.toString)
    override def addStopWordsRemover(inputColumn: String, outputColumn: String): RandomForestModelBuilder = {
      this.remover = new StopWordsRemover()
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)

      this.filteredDataFrame = remover.transform(this.tokenizedDataFrame)

      this
    }

    //.removeColumn(SparkUtil.DataColumns.tokens.toString)
    override def removeColumn(columnName: String): RandomForestModelBuilder = {
      this.finalDataFrame = filteredDataFrame.drop(columnName)

      this
    }

    //.addLabelIndexer(SparkUtil.DataColumns.label.toString, SparkUtil.DataColumns.indexedLabel.toString, "skip")
    override def addLabelIndexer(inputColumn: String, outputColumn: String, handleInvalid: String): RandomForestModelBuilder = {
      this.labelIndexer = new StringIndexer()
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
        .setHandleInvalid(handleInvalid) // skip: Parameter for how to handle invalid data (unseen/null values)
        .fit(this.finalDataFrame)

      this
    }

    //.addCountVectorizer(SparkUtil.DataColumns.filtered.toString, SparkUtil.DataColumns.features.toString, Predef.int2Integer(1000))
    override def addCountVectorizer(inputColumn: String, outputColumn: String, vocabularySize: Integer): RandomForestModelBuilder = {
      this.vectorizer = new CountVectorizer()
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
        .setVocabSize(Predef.Integer2int(vocabularySize)) //1000 vocab size
        .fit(this.finalDataFrame)

      this
    }

    override def applyRandomSplit(weights: Array[Double], seed: Long): RandomForestModelBuilder = {
      val Array(trainDF, testDF) =
        this.finalDataFrame.randomSplit(weights, seed) //activeDataFrame.randomSplit(Array(.8, .2), 1)

      this.trainDataFrame = trainDF
      this.testDataFrame = testDF

      this
    }

    override def convertLabelsToString(inputColumn: String, outputColumn: String): RandomForestModelBuilder = {
      this.labelConverter = new IndexToString()
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
        .setLabels(labelIndexer.labels)

      this
    }

    override def addRandomForestClassifier(labelColumn: String, featuresColumn: String): RandomForestModelBuilder = {
      this.randomForestClassifier = new RandomForestClassifier()
        .setLabelCol(labelColumn)
        .setFeaturesCol(featuresColumn)

      this
    }

    override def addParameterGrid(maxDepth: Array[Int], numberOfTrees: Array[Int]): RandomForestModelBuilder = {
      this.parameterGrid = new ParamGridBuilder()
        .addGrid(this.randomForestClassifier.maxDepth, maxDepth) //Array(2, 5, 8, 15)  Array(2, 5) -> build random forest depth 2 and build random forest depth 5.....
        .addGrid(this.randomForestClassifier.numTrees, numberOfTrees) //Array(200, 200) Array(6, 12)
        .build()

      this
    }

    override def addMulticlassClassificationEvaluator(labelColumn: String, predictionColumn: String, metricName: String): RandomForestModelBuilder = {
      this.multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(labelColumn)
        .setPredictionCol(predictionColumn)
        .setMetricName(metricName) //accuracy

      this
    }

    override def addCrossValidator(numberOfFolds: Integer): RandomForestModelBuilder = {
      this.crossValidator = new CrossValidator()
        .setEstimator(this.randomForestClassifier)
        .setEvaluator(this.multiclassClassificationEvaluator)
        .setEstimatorParamMaps(this.parameterGrid)
        .setNumFolds(Predef.Integer2int(numberOfFolds))

      this
    }

    override def addPipelineStages(): RandomForestModelBuilder = {
      this.pipeline = new Pipeline().setStages(Array(this.labelIndexer
        , this.vectorizer, this.crossValidator, this.labelConverter))

      this
    }

    //    override def createPipelineModel: RandomForestModelBuilder = {
    //
    //    }

    override def createPredictions: RandomForestModelBuilder = {
      this.pipelineModel = pipeline.fit(this.trainDataFrame)
      this.predictionsDataFrame = pipelineModel.transform(this.testDataFrame)

      this
    }

    override def getAccuracy: RandomForestModelBuilder = {
      this.accuracy = this.multiclassClassificationEvaluator.evaluate(this.predictionsDataFrame)
      this
    }

    override def getModelWithBestHyperConfig: RandomForestModelBuilder = {
      val cvModel = pipelineModel.stages(2).asInstanceOf[CrossValidatorModel]
      this.modelWithBestHyperParameterConfig = cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

      this
    }

    override def build: RandomForest = new RandomForestPrediction(this)
  }

}
