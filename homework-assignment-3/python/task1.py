"""
Homework Assignment 3
Task 1

Locality Sensitive Hashing on signature matrix computed by min-hashing user / business review data to identify similar businesses.

Representation: Businesses (sets) are represented in the dimension of users (elements).
"""

import os
import sys

from collections import namedtuple
from datetime import datetime
from itertools import chain, combinations
from pyspark import SparkConf, SparkContext
from random import randint


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'hw3-task1'
    run_time_params['in_file'] = sys.argv[1]
    run_time_params['out_file'] = sys.argv[2]
    run_time_params['bands'] = 25
    run_time_params['n_hashers'] = 50
    run_time_params['jaccard_threshold'] = 0.5
    run_time_params['prime_modulus'] = 4213398913  # randomly generated prime in range 1 billion to 10 billion
    return run_time_params


def parse_dataset():
    with open(params['in_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['in_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], record[0])) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], set(business_set[1])))


def get_element_mapping(dataset_rdd):
    # business_ids = dataset_rdd.map(lambda business_set: business_set[0]).collect()
    user_ids = list(chain.from_iterable(dataset_rdd.map(lambda business_set: business_set[1]).collect()))

    def map_by_counter(ids, _map, counter):
        for _id in ids:
            if _id not in _map:
                _map[_id] = counter
                counter += 1
        return _map

    return map_by_counter(user_ids, dict(), 0)  # , map_by_counter(business_ids, dict(), 0)


def construct_hashers(num_rows):
    HasherParams = namedtuple('HasherParams', ['a', 'b', 'm'])
    hasher_params_list = [
        HasherParams(a=randint(1, sys.maxsize), b=randint(0, sys.maxsize), m=num_rows)
        for _ in range(params['n_hashers'])
    ]

    # https://stackoverflow.com/a/25104050/16112875
    return list(map(
        lambda hash_params:
        lambda x:
        ((hash_params.a * x + hash_params.b) % params['prime_modulus']) % hash_params.m,
        hasher_params_list
    ))


def get_signature(dataset_rdd, hashers, element_map):
    def _find_min_sig(id_set_pair):
        _id = id_set_pair[0]
        _set = id_set_pair[1]

        sig_vector = list()
        for hasher in hashers:
            sig_val = sys.maxsize
            for element in _set:
                ele_val = hasher(element_map[element])
                sig_val = min(sig_val, ele_val)
            sig_vector.append(sig_val)
        return _id, sig_vector

    return dataset_rdd.map(_find_min_sig)


def get_band_wise_rdd(signature):
    r = params['n_hashers'] // params['bands']
    return signature \
        .map(
            lambda business_set: (
                business_set[0],
                list(map(
                    lambda chunk_num: tuple(business_set[1][chunk_num * r: chunk_num * r + r]),
                    range(params['bands'])
                ))
            )
        )


def get_candidate_rdd(band_wise_rdd):
    return band_wise_rdd \
        .flatMapValues(lambda _set: [(band_num, band) for band_num, band in enumerate(_set)]) \
        .map(lambda business_band: (business_band[1], business_band[0])) \
        .groupByKey() \
        .map(lambda chunk_wise_candidates: chunk_wise_candidates[1]) \
        .filter(lambda candidate_list: len(candidate_list) > 1) \
        .flatMap(lambda candidate_list: [tuple(sorted(pair)) for pair in combinations(candidate_list, 2)])


def get_jaccard_similarity(user_set_1, user_set_2):
    return len(user_set_1.intersection(user_set_2)) / len(user_set_1.union(user_set_2))


def get_similar_pairs(candidate_pairs, dataset_map):
    return list(
        filter(
            lambda res_pair: res_pair[2] >= params['jaccard_threshold'],
            map(
                lambda pair: (
                    pair[0],
                    pair[1],
                    get_jaccard_similarity(
                        dataset_map[pair[0]],
                        dataset_map[pair[1]]
                    )),
                candidate_pairs
            )
        )
    )


def write_results_to_file(similar_pairs):
    file_header = 'business_id_1, business_id_2, similarity\n'
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for pair in similar_pairs:
            fh.write('{},{},{}\n'.format(pair[0], pair[1], pair[2]))


def main():
    # dataset rdd
    dataset_rdd = parse_dataset()

    # get element (row / users) and set (cols / businesses) mappings
    element_map = get_element_mapping(dataset_rdd)

    # define a collection of hash functions
    hashers = construct_hashers(len(element_map))

    # construct a column-wise signature matrix
    signature = get_signature(dataset_rdd, hashers, element_map)

    # chunk each column into bands
    band_wise_rdd = get_band_wise_rdd(signature)

    # find candidate pairs
    candidate_pairs = sorted(set(get_candidate_rdd(band_wise_rdd).collect()))

    # get actual similar pairs from candidate pairs
    similar_pairs = get_similar_pairs(candidate_pairs, dataset_rdd.collectAsMap())
    write_results_to_file(similar_pairs)


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run min-hashing + locality sensitive hashing
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
