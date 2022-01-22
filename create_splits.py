import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 2 sub folders: train / val
    """
    # TODO: Implement function
    source = os.path.join(source, '*.tfrecord')
    train_path = os.path.join(destination, 'train')
    val_path = os.path.join(destination, 'val')

    filenames = []
    for filename in glob.iglob(source, recursive = True):
        filenames.append(filename)
    
    num_files = len(filenames)
    train_indices = random.sample(range(num_files), int(num_files * 0.85))

    for idx in range(num_files):
        filename = os.path.basename(filenames[idx])
        if idx in train_indices:
            shutil.move(filenames[idx], os.path.join(train_path, filename))
        else:
            shutil.move(filenames[idx], os.path.join(val_path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)