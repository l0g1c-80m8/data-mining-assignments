"""
Homework Assignment 1
Task 1

Read data from the input file and run analytics on it to answer the required queries
"""

import json
import os
import sys

from operator import add
from pyspark import SparkConf, SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) < 3:
    # expected arguments: script path, dataset path, output file path
    print('ERR: Expected three arguments: script path, dataset path and output file path.')
    exit(1)

# read program arguments
dataset_path = sys.argv[1]
result_path = sys.argv[2]

# create an object to store the results
results = {}

# set spark app config
sc = SparkContext(conf=SparkConf().setAppName("hw1-task1").setMaster("local[*]"))
sc.setLogLevel('WARN')

# read the data and construct a spark RDD object
datasetRDD = sc.textFile(dataset_path)

# convert each text line into json objects and cache the RDD for processing
datasetRDD = datasetRDD.map(lambda rawLine: json.loads(rawLine)).cache()

# [Query] A. The total number of reviews (0.5 point)
results['n_review'] = datasetRDD.count()

# [Query] B. The number of reviews in 2018 (0.5 point)
results['n_review_2018'] = datasetRDD.filter(lambda reviewObj: reviewObj['date'].startswith('2018-')).count()

# [Query] C. The number of distinct users who wrote reviews (0.5 point)
results['n_user'] = datasetRDD.map(lambda reviewObj: reviewObj['user_id']).distinct().count()

# [Query] D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote (0.5 point)
# https://stackoverflow.com/questions/48703081/pyspark-takeordered-multiple-fields-ascending-and-descending
results['top10_user'] = datasetRDD \
    .map(lambda reviewObj: (reviewObj['user_id'], 1)) \
    .reduceByKey(add) \
    .takeOrdered(10, key=lambda user_count: [-user_count[1], user_count[0]])

# [Query] E. The number of distinct businesses that have been reviewed (0.5 point)
results['n_business'] = datasetRDD.map(lambda reviewObj: reviewObj['business_id']).distinct().count()

# [Query] F. The top 10 businesses that had the largest number of reviews and the number of reviews they had (0.5 point)
results['top10_business'] = datasetRDD \
    .map(lambda reviewObj: (reviewObj['business_id'], 1)) \
    .reduceByKey(add) \
    .takeOrdered(10, key=lambda business_count: [-business_count[1], business_count])

# print the results on console for local execution
if len(sys.argv) > 3 and sys.argv[3] == '--local':
    print('Computed results:', results)

# write results to the file at result_path
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
with open(result_path, 'w') as file_handle:
    json.dump(results, file_handle)
