"""
Homework Assignment 2
Task 2

Read data from the Ta-Feng dataset, pre-process it and implement the SON algorithm
with Apriori algorithm to fund frequent item-sets.
"""

import os
import sys

from collections import defaultdict, Counter
from datetime import datetime
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
    params['app_name'] = 'hw2-task2'
    params['threshold'] = int(sys.argv[1])
    params['support'] = int(sys.argv[2])
    params['in_file'] = sys.argv[3]
    params['out_file'] = sys.argv[4]
    params['processed_out_file'] = './processed.csv'
    return params


def write_csv(header_list, rows):
    with open(params['processed_out_file'], 'w') as file_handle:
        file_handle.write(','.join(header_list))
        file_handle.write('\n')
        for row in rows:
            file_handle.write(','.join(row))
            file_handle.write('\n')


def combine_ids_from_row(data_row, header_dict):
    row_transaction_date = data_row[header_dict["TRANSACTION_DT"]]
    row_transaction_date = row_transaction_date[: -4] + row_transaction_date[-2:]
    row_customer_id = data_row[header_dict["CUSTOMER_ID"]]
    row_product_id = str(int(data_row[header_dict["PRODUCT_ID"]]))

    combined_id = '{}-{}'.format(row_transaction_date, row_customer_id)
    return combined_id, row_product_id


def get_header_dict(file_name, encoding='utf-8'):
    header_dict = defaultdict(int)
    file_header_row = []
    with open(file_name, 'r', encoding=encoding) as file_handle:
        file_header_row = file_handle.readline().replace('"', '').strip().split(',')
        for idx, header in enumerate(file_header_row):
            header_dict[header] = idx
    return header_dict, file_header_row


def parse_raw_data():
    header_dict, file_header_row = get_header_dict(params['in_file'], 'utf-8-sig')
    raw_data_rdd = sc.textFile(params['in_file']) \
        .map(lambda line: line.replace('"', '').strip().split(',')) \
        .filter(lambda line: line != file_header_row) \
        .map(lambda line: combine_ids_from_row(line, header_dict))
    write_csv(['DATE-CUSTOMER_ID', 'PRODUCT_ID'], raw_data_rdd.collect(), )


def parse_processed_data():
    header_dict, file_header_row = get_header_dict(params['processed_out_file'], 'utf-8')
    file_header_tuple = tuple(file_header_row)
    processed_data_rdd = sc.textFile(params['processed_out_file']) \
        .map(lambda line: tuple(line.split(','))) \
        .filter(lambda line: line != file_header_tuple)
    return processed_data_rdd


def filter_processed_data():
    processed_data_rdd = processed_rdd.groupByKey()
    filtered_data_rdd = processed_data_rdd \
        .filter(lambda id_list: len(id_list[1]) > params['threshold']) \
        .map(lambda id_list: (id_list[0], set(id_list[1])))
    return filtered_data_rdd


def write_item_sets_by_count(item_sets_by_size, header, mode='w'):
    with open(params['out_file'], mode) as file_handle:
        file_handle.write(header)
        file_handle.write('\n')
        for size, candidates in item_sets_by_size.items():
            if len(candidates) == 0:
                file_handle.write('\n\n')
            if size == 1:
                for candidate in candidates[: -1]:
                    file_handle.write('(\'{}\'),'.format(candidate))
                file_handle.write('(\'{}\')\n\n'.format(candidates[-1]))
            else:
                for candidate in candidates[: -1]:
                    file_handle.write('{},'.format(candidate))
                file_handle.write(str(candidates[-1]))
                file_handle.write('\n\n')


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
    # find candidates from individual chunks - first phase
    candidates_rdd = filtered_rdd.mapPartitions(apriori)
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
        filtered_rdd.mapPartitions(lambda chunk: get_frequents(chunk, candidates_by_size)).reduceByKey(add).collect()
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

# Part A. parse raw data and generate the intermediate file
parse_raw_data()

# Part B. use the intermediate file, filter the counts and apply SON + Apriori
processed_rdd = parse_processed_data()
# filter the rdd tp get only those transactions that meet the filter threshold
filtered_rdd = filter_processed_data()
total_transaction_count = filtered_rdd.count()

# run SON with Apriori
start_ts = datetime.now()
execute_son()
end_ts = datetime.now()
print('Duration: ', (end_ts - start_ts).total_seconds())

# exit without errors
exit(0)
