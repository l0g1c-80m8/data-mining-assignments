"""
Homework Assignment 5
Task 1

Implement a bloom filter on a simulated datastream.
"""


import sys

from argparse import Namespace


def get_runtime_params():
    if len(sys.argv) < 5:
        print('ERR: Expected four arguments: (i/p file, stream_size, num_of_asks, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        INPUT_FILE=sys.argv[1],
        STREAM_SIZE=sys.argv[2],
        NUM_OF_ASKS=sys.argv[3],
        Output_FILE=sys.argv[4],
    )


if __name__ == '__main__':
    # get runtime params
    PARAMS = get_runtime_params()
    print(PARAMS)
