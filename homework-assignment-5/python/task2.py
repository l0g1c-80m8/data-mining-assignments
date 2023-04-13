"""
Homework Assignment 5
Task 2

Implement Flajolet-Martin algorithm to estimate the number of unique users in a window of data steam.
"""

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
        lambda x: format(
            ((hash_params.a * x + hash_params.b) % PARAMS.PRIME_MODULUS) % hash_params.m,
            '0{}b'.format(PARAMS.BIN_HASH_LEN)
        ),
        reduce(
            lambda hashers, _: hashers + [
                HasherParams(a=randint(1, sys.maxsize), b=randint(0, sys.maxsize), m=randint(0, sys.maxsize))
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
        APP_NAME='hw5-task2',
        INPUT_FILE=sys.argv[1],
        STREAM_SIZE=int(sys.argv[2]),
        NUM_OF_ASKS=int(sys.argv[3]),
        OUTPUT_FILE=sys.argv[4],
        IS_DEBUG=sys.argv[-1] == '--debug',
        N_HASHERS=30,
        CHUNK_SIZE=2,
        BIN_HASH_LEN=64,
        PRIME_MODULUS=4213398913,  # randomly generated prime in range 1 billion to 10 billion
        OUT_FILE_HEADERS=['Time', 'Ground Truth', 'Estimation']
    )


def get_runtime_global_ns():
    return Namespace(
        BX=BlackBox(),
        HASHERS=construct_hashers()
    )


def myhashs(user):
    """
    The purpose of this util is to expose the results of the get_user_hashes with a new name for external imports,
    so that the results can be evaluated
    :param user:
    :return: user hashed indices on filter array
    """
    return get_user_hashes(user)


def get_user_hashes(user):
    user_int_hash = int(hexlify(user.encode('utf8')), 16)
    return reduce(
        lambda user_hashes, hash_func: user_hashes + [hash_func(user_int_hash)],
        GLOB_NS.HASHERS,
        list()
    )


def write_results_to_file(results):
    with open(PARAMS.OUTPUT_FILE, 'w') as fh:
        fh.write('{}\n'.format(','.join(PARAMS.OUT_FILE_HEADERS)))
        for result in results:
            fh.write('{},{},{}\n'.format(result[0], result[1], result[2]))


def count_trailing_zeroes(binary_num):
    binary_string = str(binary_num)
    return len(binary_string) - len(binary_string.rstrip('0'))


def debug_perf(results):
    return reduce(
        lambda acc, result: acc + result[2],
        results,
        0
    ) / reduce(
        lambda acc, result: acc + result[1],
        results,
        0
    )


def main():
    results = list()
    for run_idx_ in range(PARAMS.NUM_OF_ASKS):
        users = set()
        max_trailing_zeros_len = [0] * PARAMS.N_HASHERS
        for user in GLOB_NS.BX.ask(PARAMS.INPUT_FILE, PARAMS.STREAM_SIZE):
            users.add(user)
            for idx, user_hash in enumerate(get_user_hashes(user)):
                max_trailing_zeros_len[idx] = max(max_trailing_zeros_len[idx], count_trailing_zeroes(user_hash))
        n_chunks = PARAMS.N_HASHERS // PARAMS.CHUNK_SIZE
        chunk_avg_trailing_zero = sorted(map(
            lambda chunk_idx: sum(
                max_trailing_zeros_len[chunk_idx * PARAMS.CHUNK_SIZE:(chunk_idx + 1) * PARAMS.CHUNK_SIZE]
            ) / PARAMS.CHUNK_SIZE,
            range(n_chunks)
        ))
        max_trailing_zero_median = chunk_avg_trailing_zero[n_chunks // 2]
        max_trailing_zero_median += chunk_avg_trailing_zero[n_chunks // 2 - 1] if n_chunks % 2 != 0 else 0
        max_trailing_zero_median /= 2 if n_chunks % 2 != 0 else 1
        max_trailing_zero_median = round(max_trailing_zero_median)
        results.append((run_idx_, len(users), 2 ** max_trailing_zero_median))

    # write the results to a file
    write_results_to_file(results)

    # process results (debug only)
    if PARAMS.IS_DEBUG:
        print(debug_perf(results))


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
