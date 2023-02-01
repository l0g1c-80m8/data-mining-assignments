# check if all arguments are received
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 1;
fi

# set correct env vars
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export PYSPARK_PYTHON=python3.6

# execute the script
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G "$1" "$2" "$3"
