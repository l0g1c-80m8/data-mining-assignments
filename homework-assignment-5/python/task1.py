"""
Homework Assignment 5
Task 1

Implement a bloom filter on a simulated datastream.
"""

import os
import sys

from argparse import Namespace
from binascii import hexlify
from blackbox import BlackBox  # local import
from collections import namedtuple
from datetime import datetime
from functools import reduce
from random import randint


def construct_hashers():
    HasherParams = namedtuple('HasherParams', ['a', 'b', 'm'])
    return list(map(
        lambda hash_params:
        lambda x:
        ((hash_params.a * x + hash_params.b) % PARAMS.PRIME_MODULUS) % hash_params.m,
        reduce(
            lambda hashers, _: hashers + [
                HasherParams(a=randint(1, sys.maxsize), b=randint(0, sys.maxsize), m=PARAMS.FILTER_ARR_LEN)
            ],
            range(PARAMS.N_HASHERS),
            list()
        )
    ))


def get_runtime_params():
    if len(sys.argv) < 5:
        print('ERR: Expected four arguments: (i/p file, stream_size, num_of_asks, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        APP_NAME='hw5-task1',
        INPUT_FILE=sys.argv[1],
        STREAM_SIZE=int(sys.argv[2]),
        NUM_OF_ASKS=int(sys.argv[3]),
        OUTPUT_FILE=sys.argv[4],
        FILTER_ARR_LEN=69997,
        N_HASHERS=10,
        PRIME_MODULUS=4213398913,  # randomly generated prime in range 1 billion to 10 billion
        OUT_FILE_HEADERS=['Time', 'FPR']
    )


def get_runtime_global_ns():
    return Namespace(
        BX=BlackBox(),
        FILTER_ARRAY=[0] * PARAMS.FILTER_ARR_LEN,
        HASHERS=construct_hashers()
    )


def get_user_hashes(user):
    user_int_hash = int(hexlify(user.encode('utf8')), 16)
    return reduce(
        lambda user_hashes, hash_func: user_hashes + [hash_func(user_int_hash)],
        GLOB_NS.HASHERS,
        list()
    )


def check_user_in_bloom(user_hash_idxes):
    return True if reduce(
        lambda acc, bloom_index: acc + GLOB_NS.FILTER_ARRAY[bloom_index],
        user_hash_idxes,
        0
    ) == PARAMS.N_HASHERS else False


def populate_bloom(user_hash_idxes):
    for bloom_index in user_hash_idxes:
        GLOB_NS.FILTER_ARRAY[bloom_index] = 1


def write_results_to_file(results):
    with open(PARAMS.OUTPUT_FILE, 'w') as fh:
        fh.write('{}\n'.format(','.join(PARAMS.OUT_FILE_HEADERS)))
        for result in results:
            fh.write('{}, {}\n'.format(result[0], result[1]))


def main():
    results = list()
    for run_idx_ in range(PARAMS.NUM_OF_ASKS):
        seen_users = set()
        false_bloom_count = 0
        for user in GLOB_NS.BX.ask(PARAMS.INPUT_FILE, PARAMS.STREAM_SIZE):
            user_hash_idxes = get_user_hashes(user)
            false_bloom_count += 1 if check_user_in_bloom(user_hash_idxes) and user not in seen_users else 0
            populate_bloom(user_hash_idxes)
            seen_users.add(user)
        fpr = false_bloom_count / len(seen_users)
        results.append((run_idx_, fpr))

    # write the results to a file
    write_results_to_file(results)


if __name__ == '__main__':
    # get runtime params
    PARAMS = get_runtime_params()

    # define a namespace for global variables
    GLOB_NS = get_runtime_global_ns()

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
