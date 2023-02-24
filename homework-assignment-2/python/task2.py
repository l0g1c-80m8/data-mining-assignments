"""
Homework Assignment 2
Task 2

Read data from the Ta-Feng dataset, pre-process it and implement the SON algorithm
with Apriori algorithm to fund frequent item-sets.
"""

import os
import sys

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
        .filter(lambda id_list: len(id_list[1]) >= params['threshold']) \
        .map(lambda id_list: (id_list[0], set(id_list[1])))
    return filtered_data_rdd


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
print(filtered_rdd.take(5))

# exit without errors
exit(0)
