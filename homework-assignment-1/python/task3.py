"""
Homework Assignment 1
Task 3

Two datasets are explored together containing review information and business information.
This task requires combining results from both datasets to arrive at the final results.
"""

import json
import os
import sys

from datetime import datetime, timedelta
from operator import add
from pyspark import SparkConf, SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) < 5:
    # expected arguments: script path, dataset path, output file path
    print('ERR: Expected three arguments: script path, dataset path and output file path.')
    exit(1)

# read program arguments
review_path = sys.argv[1]
business_path = sys.argv[2]
resultA_path = sys.argv[3]
resultB_path = sys.argv[4]

# set spark app config
sc = SparkContext(conf=SparkConf().setAppName("hw1-task2").setMaster("local[*]"))
sc.setLogLevel('WARN')

'''
Part A. Compute the average stars for each city using the stars in review.json and city names in business.json
'''

# read the data and construct spark RDDs and map them to a form that supports join between the two
reviewRDD = sc.textFile(review_path) \
    .map(lambda raw: json.loads(raw)) \
    .map(lambda review: (review['business_id'], review['stars']))
businessRDD = sc.textFile(business_path) \
    .map(lambda raw: json.loads(raw)) \
    .map(lambda business: (business['business_id'], business['city']))

# join the two RDDs using business_id as key
combinedRDD = reviewRDD.join(businessRDD)

# map the RDD to a form where stars are reducible by city and reduce the value
combinedRDD = combinedRDD \
    .map(lambda entry: (entry[1][1], (entry[1][0], 1))) \
    .reduceByKey(lambda count_sum1, count_sum2: (count_sum1[0] + count_sum2[0], count_sum1[1] + count_sum2[1])) \
    .map(lambda entry: (entry[0], entry[1][0] / entry[1][1])) \
    .sortBy(lambda entry: [-entry[1], entry[0]]) \
    .collect()

'''
Part B. Compare the results for the same tasks using different methods and reason about it
'''

# create output object for results of task B
resultsB = {
    'm1': 0,
    'm2': 0,
    'reason': 'default'
}

# print the results on console for local execution
if sys.argv[len(sys.argv) - 1] == '--local':
    print('Computed results:', '\nPartA.\n', combinedRDD, '\nPartB.\n', resultsB)

# write results for part A to a text file at resultA_path
# https://www.pythontutorial.net/python-basics/python-write-text-file/
with open(resultA_path, 'w') as file_handle:
    file_handle.write('city,stars\n')
    for entry in combinedRDD:
        file_handle.write(str(entry[0]) + ',' + str(entry[1]) + '\n')

# write results for part B to the file at resultB_path
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
with open(resultB_path, 'w') as file_handle:
    json.dump(resultsB, file_handle)

# exit without errors
exit(0)
