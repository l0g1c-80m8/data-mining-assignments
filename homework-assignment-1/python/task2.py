"""
Homework Assignment 1
Task 2

Using a custom partitioning of data to improve the performance of the map-reduce computation
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
    # expected arguments: script path, dataset path, output file path
    print('ERR: Expected three arguments: script path, dataset path and output file path.')
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
RDD construction using default partition scheme
'''
# convert each text line into json objects and cache the RDD for processing
defaultDatasetRDD = datasetRDD.map(lambda rawLine: json.loads(rawLine))

# Task 1 Query F. The top 10 businesses that had the largest number of reviews and the number of reviews they had
# https://docs.python.org/3.6/library/datetime.html

# first clock measurement
ts1 = datetime.now()

defaultDatasetRDD.map(lambda reviewObj: (reviewObj['business_id'], 1)) \
    .partitionBy(n_partition) \
    .reduceByKey(add) \
    .takeOrdered(10, key=lambda business_count: [-business_count[1], business_count])
# second clock measurement
ts2 = datetime.now()

# store results
results['default'] = {
    'n_partition': defaultDatasetRDD.getNumPartitions(),
    'n_items': defaultDatasetRDD.glom().map(lambda partition_list: len(partition_list)).collect(),
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
