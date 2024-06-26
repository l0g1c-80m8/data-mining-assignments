"""
Homework Assignment 1
Task 2

Using a custom partitioning of data to improve the performance of the map-reduce computation.

Computation description:
Task 1 Query F: The top 10 businesses that had the largest number of reviews and the number of reviews they had.
"""

import json
import os
import sys

from datetime import datetime, timedelta
from operator import add
from pyspark import SparkConf, SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) < 4:
    # expected arguments: script path, dataset path, output file path, number of partitions
    print('ERR: Expected three arguments: script path, dataset path, output file path and number of partitions.')
    exit(1)

# read program arguments
dataset_path = sys.argv[1]
result_path = sys.argv[2]
n_partition = int(sys.argv[3])

# initialize an object to store the results
results = {}

# set spark app config
sc = SparkContext(conf=SparkConf().setAppName("hw1-task2").setMaster("local[*]"))
sc.setLogLevel('WARN')

# read the data and construct a spark RDD object
datasetRDD = sc.textFile(dataset_path)

'''
Part 1. RDD construction using default partition scheme
'''

# convert each text line into json objects and cache the RDD for processing
defaultDatasetRDD = datasetRDD.map(lambda rawLine: json.loads(rawLine))

# first clock measurement
ts1 = datetime.now()

defaultDatasetRDD.map(lambda reviewObj: (reviewObj['business_id'], 1)) \
    .reduceByKey(add) \
    .takeOrdered(10, key=lambda business_count: [-business_count[1], business_count])

# second clock measurement
ts2 = datetime.now()

# store results
results['default'] = {
    'n_partition': defaultDatasetRDD.getNumPartitions(),
    'n_items': defaultDatasetRDD.glom().map(len).collect(),
    'exe_time': (ts2 - ts1) / timedelta(microseconds=1),  # https://docs.python.org/3.6/library/datetime.html
}

'''
Part 2. RDD construction using custom partition scheme

Reference: https://www.talend.com/resources/intro-apache-spark-partitioning/
'''

# convert each text line into json objects and cache the RDD for processing
customDatasetRDD = datasetRDD.map(lambda rawLine: json.loads(rawLine)).map(lambda reviewObj: (reviewObj['business_id'], 1)) \

# first clock measurement
ts1 = datetime.now()

customDatasetRDD = customDatasetRDD.partitionBy(n_partition, lambda business_id: ord(business_id[-1]) % n_partition)
customDatasetRDD.reduceByKey(add).takeOrdered(10, key=lambda business_count: [-business_count[1], business_count])

# second clock measurement
ts2 = datetime.now()

# store results
results['customized'] = {
    'n_partition': customDatasetRDD.getNumPartitions(),
    'n_items': customDatasetRDD.glom().map(len).collect(),
    'exe_time': (ts2 - ts1) / timedelta(microseconds=1),
}

# print the results on console for local execution
if sys.argv[len(sys.argv) - 1] == '--local':
    print('Computed results:', results)

# write results to the file at result_path
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
with open(result_path, 'w') as file_handle:
    json.dump(results, file_handle)

# exit without errors
exit(0)
