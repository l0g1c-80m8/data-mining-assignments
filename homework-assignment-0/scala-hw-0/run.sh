# set correct environments for scala
export SPARK_HOME=/Library/spark-3.1.2-bin-hadoop3.2
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home

# submit job to spark
/Library/spark-3.3.1-bin-hadoop3/bin/spark-submit "$@"