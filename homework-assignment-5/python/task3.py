"""
Homework Assignment 5
Task 3

Implement reservoir sampling
"""

import sys

from argparse import Namespace
from blackbox import BlackBox  # local import
from datetime import datetime
from random import randint, random, seed


def get_runtime_params():
    if len(sys.argv) < 5:
        print('ERR: Expected four arguments: (i/p file, stream_size, num_of_asks, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        APP_NAME='hw5-task3',
        INPUT_FILE=sys.argv[1],
        STREAM_SIZE=int(sys.argv[2]),
        NUM_OF_ASKS=int(sys.argv[3]),
        OUTPUT_FILE=sys.argv[4],
        RES_SIZE=100,
        PRNG_SEED=553,
        OUT_FILE_HEADERS=['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id']
    )


def write_results_to_file(results):
    with open(PARAMS.OUTPUT_FILE, 'w') as fh:
        fh.write('{}\n'.format(','.join(PARAMS.OUT_FILE_HEADERS)))
        for result in results:
            fh.write('{},{}\n'.format(result[0], ','.join(result[1])))


def main():
    # set random seed
    seed(PARAMS.PRNG_SEED)

    # start processing the random stream
    results = list()
    reservoir = list()
    for run_idx in range(PARAMS.NUM_OF_ASKS):
        for user_idx, user in enumerate(bx.ask(PARAMS.INPUT_FILE, PARAMS.STREAM_SIZE)):
            if len(reservoir) < PARAMS.RES_SIZE:
                reservoir.append(user)
            else:
                if random() < PARAMS.RES_SIZE / (run_idx * PARAMS.RES_SIZE + user_idx + 1):
                    reservoir[randint(0, PARAMS.RES_SIZE - 1)] = user
            if (run_idx * PARAMS.RES_SIZE + user_idx + 1) % PARAMS.RES_SIZE == 0:
                results.append((PARAMS.RES_SIZE * (run_idx + 1), reservoir[0::20]))

    write_results_to_file(results)


if __name__ == '__main__':
    # get runtime params
    PARAMS = get_runtime_params()

    # initialize the blackbox stream object
    bx = BlackBox()

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
