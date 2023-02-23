"""
Homework Assignment 2
Task 1

Read data from the input file and implement SON algorithm with PCY algorithm to fund frequent item-sets.
"""

import os
import sys

from collections import defaultdict, Counter
from datetime import datetime, timedelta
from functools import reduce
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
        candidates = map(lambda pair: tuple(sorted(pair)), combinations(chunk_frequent_prev, 2))
    if frequent_item_set_size > 2:
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
                frequents.append(candidate)
                break

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


def write_item_sets_by_count(item_sets_by_size, header, mode='w'):
    with open(params['out_file'], mode) as file_handle:
        file_handle.write(header)
        file_handle.write('\n')
        for size, candidates in item_sets_by_size.items():
            if size == 1:
                for candidate in candidates[: -1]:
                    file_handle.write('(\'{}\'),'.format(candidate))
                file_handle.write('(\'{}\')\n\n'.format(candidates[-1]))
            else:
                for candidate in candidates[: -1]:
                    file_handle.write('{},'.format(candidate))
                file_handle.write(str(candidates[-1]))
                file_handle.write('\n\n')


def get_frequents(chunk, candidates_by_size):
    candidates_by_count = defaultdict(int)
    candidate_counts = []
    # create a list of item sets from transaction chunk/partition
    transactions = list(map(lambda transaction: transaction[1], chunk))
    # calculate the adjusted support for this chunk/partition
    chunk_support = params['support'] * len(transactions) / total_transaction_count

    for size, candidate_item_sets in candidates_by_size.items():
        for candidate_item_set in candidate_item_sets:
            for transaction in transactions:
                item_set = set(candidate_item_set) if type(candidate_item_set) == tuple else {candidate_item_set}
                if set(item_set).issubset(transaction):
                    candidates_by_count[candidate_item_set] += 1
    for candidate_item_set, count in candidates_by_count.items():
        candidate_counts.append((candidate_item_set, count))

    return candidate_counts


def execute_son():
    # get the rdd depending on the params
    transaction_rdd = construct_rdd()
    # find candidates from individual chunks - first phase
    candidates_rdd = transaction_rdd.mapPartitions(apriori)
    candidates = candidates_rdd \
        .groupByKey() \
        .sortBy(lambda freq_set: freq_set[0]) \
        .collect()
    candidates_by_size = defaultdict(list)
    for size, candidate_item_sets in candidates:
        candidates_by_size[size] = sorted(set(reduce(lambda lis1, lis2: lis1 + lis2, candidate_item_sets, list())))
    # print the candidates
    write_item_sets_by_count(candidates_by_size, 'Candidates:', 'w')

    # find the truly frequent item sets - eliminate false positives - second phase
    frequents = list(filter(
        lambda item_set_count: item_set_count[1] >= params['support'],
        transaction_rdd.mapPartitions(lambda chunk: get_frequents(chunk, candidates_by_size)).reduceByKey(add).collect()
    ))
    frequents_by_size = defaultdict(list)
    for frequent_item_set, _ in frequents:
        if type(frequent_item_set) == tuple:
            frequents_by_size[len(frequent_item_set)].append(frequent_item_set)
        else:
            frequents_by_size[1].append(frequent_item_set)
    for frequent_item_set in frequents_by_size:
        frequents_by_size[frequent_item_set] = sorted(frequents_by_size[frequent_item_set])
    write_item_sets_by_count(frequents_by_size, 'Frequent Itemsets:', 'a')


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
start_ts = datetime.now()
execute_son()
end_ts = datetime.now()
print('Duration: ', (end_ts - start_ts).total_seconds())

# exit without errors
exit(0)
