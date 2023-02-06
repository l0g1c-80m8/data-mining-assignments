package org.rpatel.dsci553_assignments

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {

  def main(arg: Array[String]): Unit = {
    val input_file_path = arg(0)

    val conf = new SparkConf().setAppName("word-count").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("OFF")
    val datasetRDD = sc.textFile(input_file_path)
    val counts = datasetRDD.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_ + _)
    counts.foreach(println)
  }
}
