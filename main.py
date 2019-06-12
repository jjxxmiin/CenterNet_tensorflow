#import tensorflow as tf
#import numpy as np
import argparse

def opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="dataset directory please")

    args = parser.parse_args()

    return args


def main():
    input_size = 512
    output_size = 128
    batch_size = 128
    lr = 5e-4
    epoch = 140

    args = opts()

if __name__ == "__main__":
    main()