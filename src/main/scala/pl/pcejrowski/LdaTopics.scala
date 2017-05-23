package pl.pcejrowski

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, LDAModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object LdaTopics {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("lda-topics")
      .getOrCreate()
    val sc = spark.sparkContext

    import spark.implicits._
    val stopWords: Array[String] = sc.textFile("src/main/resources/stopwords.txt").collect().flatMap(_.stripMargin.split("\\s+"))
    val source: DataFrame = sc.textFile("src/main/resources/corpus.txt").toDF("docs")
    val tokenizer: RegexTokenizer = new RegexTokenizer().setInputCol("docs").setOutputCol("rawTokens")
    val stopWordsRemover: StopWordsRemover = new StopWordsRemover().setInputCol("rawTokens").setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ stopWords)
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setVocabSize(10000)
      .setInputCol("tokens")
      .setOutputCol("features")
    val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model: PipelineModel = pipeline.fit(source)
    val corpus: RDD[(Long, Vector)] = model.transform(source)
      .select("features")
      .rdd
      .map { case Row(features: MLVector) => Vectors.fromML(features) }
      .zipWithIndex()
      .map(_.swap)
    val vocabArray: Array[String] = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val actualNumTokens: Long = corpus.map(_._2.numActives).sum().toLong
    println(s"actualNumTokens: $actualNumTokens")

    val lda = new LDA()
    val optimizer = new EMLDAOptimizer

    lda.setOptimizer(optimizer)
      .setK(10)
      .setMaxIterations(100)

    val ldaModel: LDAModel = lda.run(corpus)


    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    val avgLogLikelihood = distLDAModel.logLikelihood / corpus.count().toDouble
    println(s"avgLogLikelihood: $avgLogLikelihood")
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    topics
      .zipWithIndex
      .foreach { case (topic, i) =>
        println(s"TOPIC $i")
        topic.foreach { case (term, weight) =>
          println(s"$term\t$weight")
        }
        println()
      }
    sc.stop()
  }
}
