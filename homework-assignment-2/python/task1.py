"""
Homework Assignment 2
Task 1

Read data from the input file and implement SON algorithm with PCY algorithm to fund frequent item-sets.
"""
import os
import sys

from collections import defaultdict, Counter
from datetime import datetime, timedelta
from itertools import chain, combinations, groupby
from operator import add
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 4:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected three arguments: (case number, support, input file path, output file path).')
        exit(1)

    # read program arguments
    params['app_name'] = 'hw2-task1'
    params['case_number'] = int(sys.argv[1])
    params['support'] = int(sys.argv[2])
    params['in_file'] = sys.argv[3]
    params['out_file'] = sys.argv[4]
    return params


def construct_rdd():
    data_rdd = sc.textFile(params['in_file']) \
        .map(lambda line: line.split(',')) \
        .filter(lambda user_business_pair: user_business_pair != ['user_id', 'business_id'])

    if params['case_number'] == 1:
        '''
        Case 1
        Using users as basket, group businesses into them as items.
        Each basket is represented by a user id and is a set of business ids as items.
        '''
        data_rdd = data_rdd.map(lambda two_list: tuple(two_list))
    elif params['case_number'] == 2:
        '''
        Case 2
        Using businesses as basket, group users into them as items.
        Each basket is represented by a business id and is a set of user ids as items.
        '''
        data_rdd = data_rdd.map(lambda two_list: tuple(reversed(two_list)))
    else:
        # No case for other values
        print('ERR: Expected case number: to be 0 or 1.')
        exit(1)

    data_rdd = data_rdd.groupByKey().mapValues(set)

    return data_rdd


def get_frequents_in_chunk(
        transactions,
        chunk_support,
        chunk_frequent_prev,
        frequent_item_set_size
):
    frequents = list()
    candidates = combinations(chunk_frequent_prev, 2)
    if frequent_item_set_size == 2:
        for candidate in candidates:
            candidate_count = 0
            for transaction in transactions:
                if set(candidate).issubset(transaction):
                    candidate_count += 1
                if candidate_count >= chunk_support:
                    frequents.append(candidate)
                    break
    else:
        candidates = filter(
            lambda candidate_set: len(candidate_set) == frequent_item_set_size,
            map(lambda candidate_pair: tuple(sorted(set(candidate_pair[0]).union(candidate_pair[1]))), candidates)
        )
        for candidate in set(candidates):
            candidate_count = 0
            for transaction in transactions:
                if set(candidate).issubset(transaction):
                    candidate_count += 1
                if candidate_count >= chunk_support:
                    frequents.append(tuple(candidate))
                    break
        print(frequents)

    return frequents


def apriori(chunk):
    # create a list of item sets from transaction chunk/partition
    transactions = list(map(lambda transaction: transaction[1], chunk))
    # calculate the adjusted support for this chunk/partition
    chunk_support = params['support'] * len(transactions) / total_transaction_count
    # initialize by calculating frequent singletons in the chunk/partition
    chunk_frequent_current = list(map(
        lambda item_count: item_count[0],
        filter(
            lambda item_count: item_count[1] >= chunk_support,
            Counter(chain.from_iterable(map(
                lambda transaction_item_set: list(transaction_item_set),
                transactions
            ))).items()
        )))
    frequent_item_set_size = 1

    # store the frequent item sets in a dictionary keyed by size
    chunk_frequents_comprehensive = defaultdict(list)
    chunk_frequents_comprehensive[frequent_item_set_size] = chunk_frequent_current

    # iteratively find frequent item sets for bugger sizes from the chunks
    while True:
        # increment frequent item set size
        frequent_item_set_size += 1
        # get chunks of the bigger size
        chunk_frequent_current = get_frequents_in_chunk(
            transactions,
            chunk_support,
            chunk_frequent_current,
            frequent_item_set_size
        )
        # quit when no more frequent item sets of the current size are found
        if len(chunk_frequent_current) == 0:
            break
        # add frequent item sets to the dictionary
        for item in chunk_frequent_current:
            chunk_frequents_comprehensive[frequent_item_set_size].append(item)

    # return the found frequent item sets of all sizes
    return list(map(lambda frequents: (frequents[0], frequents[1]), chunk_frequents_comprehensive.items()))


def execute_son():
    # get the rdd depending on the params
    transaction_rdd = construct_rdd()
    # find candidates from individual chunks
    candidates = transaction_rdd.mapPartitions(apriori)
    frequents = candidates.collect()
    print(frequents)


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = dict()
    parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # global vars
    rdd = construct_rdd()
    total_transaction_count = rdd.count()
    frequent_item_sets = list()

    # run SON with Apriori
    execute_son()
