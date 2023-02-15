"""
Homework Assignment 2
Task 1

Read data from the input file and implement SON algorithm with PCY algorithm to fund frequent item-sets.
"""

import os
import sys

from collections import defaultdict
from datetime import datetime, timedelta
from operator import add
from pyspark import SparkConf, SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# first clock measurement
start_time_stamp = datetime.now()

# ------------------------------------------------ start ------------------------------------------------ #

if len(sys.argv) < 4:
    # expected arguments: script path, dataset path, output file path
    print('ERR: Expected three arguments: (case number, support, input file path, output file path).')
    exit(1)

# read program arguments
case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

# set spark app config
sc = SparkContext(conf=SparkConf().setAppName("hw1-task2").setMaster("local[*]"))
sc.setLogLevel('ERROR')

'''
SON Algorithm

First pass:
- a. Divide the data into chunks (done earlier) (map task).
- b. Treat each chunk as a sample and run the A-Priori algorithm with threshold ps (map task) 
     (p => fraction in the chunk, s => support).
- c. Collect all item-sets found frequent in step b in any of the chunk.
     These form the candidate item-sets for the second pass (filter - map task). 

Second pass:
- d. Go through the full data-set and count the occurrence of candidate item-sets from c. of first pass and
     count the truly frequent item-sets in the whole dataset (remove false negatives) (reduce task).
'''

# [SON a.] read the data and construct a spark RDD object with appropriate structure for processing
dataRDD = sc.textFile(input_file_path) \
    .map(lambda line: line.split(',')) \
    .filter(lambda user_business_pair: user_business_pair != ['user_id', 'business_id'])

# [SON a.] convert rdd to tuple and set the key based on the case number
if case_number == 1:
    '''
    Case 1
    Using users as basket, group businesses into them as items.
    Each basket is represented by a user id and is a set of business ids as items.
    '''
    dataRDD = dataRDD.map(lambda two_list: tuple(two_list))
elif case_number == 2:
    '''
        Case 2
        Using businesses as basket, group users into them as items.
        Each basket is represented by a business id and is a set of user ids as items.
        '''
    dataRDD = dataRDD.map(lambda two_list: tuple(reversed(two_list)))
else:
    # No case for other values
    print('ERR: Expected case number: to be 0 or 1.')
    exit(1)

# [SON a.] create baskets based on the id with unique entries (list of [transaction id: set of item ids])
dataRDD = dataRDD.groupByKey().mapValues(set)
# [SON a.] get the total transactions in the RDD
total_transaction_count = dataRDD.count()


# [SON b.] define a function to execute the apriori algorithm on an RDD partition (chunk)
def apriori(chunk):
    """
    The A-Priori algorithm
    - a. initialize Ck (item-sets and their counts - given as input) for finding item-sets of size k.
    - b. examine the counts of the items to determine which of them are frequent (count >= support).
    - c. filter items not satisfying b. and return the output as Lk.

    :param chunk: the chunk to be processed
    :return: the frequent item-sets in this partition (chunk) which are candidates for the global dataset
    """
    chunk_transaction_count = 0
    chunk_item_count_dict = defaultdict(int)

    for transaction in chunk:
        chunk_transaction_count += 1
        for item in transaction[1]:
            chunk_item_count_dict[item] += 1

    # [Apriori b.] calculate the fraction of items 'p' and the support 's' for this chunk
    chunk_fraction = chunk_transaction_count / total_transaction_count
    chunk_support = chunk_fraction * support
    # [Apriori b. and c.] filter the frequent item-sets based on the support and return the results
    chunk_frequent_items = filter(lambda pair: pair[1] > chunk_support, chunk_item_count_dict.items())
    return map(lambda pair: pair[0], chunk_frequent_items)


# [SON b. and c.] run the apriori algorithm on each chunk with appropriate parameters
candidate_item_sets = dataRDD.mapPartitions(apriori).distinct(1).collect()
# [SON d.] count the truly frequent items in the whole data set
item_sets_counts = dict(dataRDD.map(lambda transaction: list(transaction[1]))
                        .flatMap(lambda item_list: item_list)
                        .map(lambda item: (item, 1))
                        .reduceByKey(add)
                        .collect())
frequent_item_sets = filter(lambda item_set: item_sets_counts[item_set] >= support, candidate_item_sets)

# ------------------------------------------------ end ------------------------------------------------ #

# second clock measurement
end_time_stamp = datetime.now()
# calculate the duration and log it to the console
print('Duration: ', (end_time_stamp - start_time_stamp) / timedelta(microseconds=1))

# exit without errors
exit(0)
