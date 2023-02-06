package hw_0

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {

  def main(arg: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("word-count").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val datasetRDD = sc.textFile("../text.txt")
    val counts = datasetRDD.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_ + _)
    counts.foreach(println)
  }
}