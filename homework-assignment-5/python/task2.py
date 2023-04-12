"""
Homework Assignment 5
Task 1

Implement Fljolet-Martin algorithm to estimate the number of unique users in a window of data steam.
"""

import sys

from argparse import Namespace
from binascii import hexlify
from blackbox import BlackBox  # local import
from datetime import datetime
from functools import reduce


def construct_hashers():
    return list()


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
        N_HASHERS=10,
        OUT_FILE_HEADERS=['Time', 'Ground Truth', 'Estimation']
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
        HASHERS,
        list()
    )


def write_results_to_file(results):
    with open(PARAMS.OUTPUT_FILE, 'w') as fh:
        fh.write('{}\n'.format(','.join(PARAMS.OUT_FILE_HEADERS)))
        for result in results:
            fh.write('{}, {}\n'.format(result[0], result[1], result[2]))


def main():
    results = list()

    # exec Flajolet-Martin here

    # write the results to a file
    write_results_to_file(results)


if __name__ == '__main__':
    # get runtime params
    PARAMS = get_runtime_params()

    # construct hashers
    HASHERS = construct_hashers()

    # init obj of BlackBox to generate streams
    bx = BlackBox()

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
