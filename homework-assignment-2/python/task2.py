"""
Homework Assignment 2
Task 2

Read data from the Ta-Feng dataset, pre-process it and implement the SON algorithm
with Apriori algorithm to fund frequent item-sets.
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
    params['app_name'] = 'hw2-task2'
    params['case_number'] = int(sys.argv[1])
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


def combine_ids(data_row, header_dict):
    row_transaction_date = data_row[header_dict["TRANSACTION_DT"]]
    row_transaction_date = row_transaction_date[: -4] + row_transaction_date[-2:]
    row_customer_id = data_row[header_dict["CUSTOMER_ID"]]
    row_product_id = str(int(data_row[header_dict["PRODUCT_ID"]]))

    combined_id = '{}-{}'.format(row_transaction_date, row_customer_id)
    return combined_id, row_product_id


def parse_raw_data():
    header_dict = defaultdict(int)
    file_headers = ''
    with open(params['in_file'], 'r', encoding='utf-8-sig') as file_handle:
        file_headers = file_handle.readline().replace('"', '').strip().split(',')
        for idx, header in enumerate(file_headers):
            header_dict[header] = idx

    raw_rdd = sc.textFile(params['in_file']) \
        .map(lambda line: line.replace('"', '').strip().split(',')) \
        .filter(lambda line: line != file_headers) \
        .map(lambda line: combine_ids(line, header_dict))
    write_csv(['DATE-CUSTOMER_ID', 'PRODUCT_ID'], raw_rdd.collect(), )


# set executables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# initialize program parameters
params = dict()
parse_args()

# create spark context
sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
sc.setLogLevel('ERROR')

# parse raw data and generate the intermediate file
parse_raw_data()

# exit without errors
exit(0)
