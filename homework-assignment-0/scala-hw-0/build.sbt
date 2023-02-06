name := "wordCount"

version := "0.1"

scalaVersion := "2.13.10"

val sparkVersion = "3.3.1"

resolvers ++= Seq(
  "apache-snapshots" at "https://reporitory.apache.org/snapshots"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion
)
