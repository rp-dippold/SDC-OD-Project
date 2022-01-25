import argparse
import glob
import os
import re
import random
import shutil
import cv2

import numpy as np

from utils import get_module_logger, get_dataset


def correct_filename(filename):
    """Remove a '_##' extension in a filenames root.

    The filenames stored in the tfrecords contain an extension '_##'
    at the end of its root part which has to be removed. Otherwise it
    is not possible to reload the data by using the filename.

    args:
    - filename str: name of the file to be corrected
    returns:
    - filename str: corrected filename
    """
    prefix, suffix = os.path.splitext(filename)
    return re.sub(r'_\d+$', suffix, prefix)


def calculate_mean_std(image_list):
    """Calculate mean and std of image list.

    args:
    - image_list [list[array]]: list of images provided as numpy arrays
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    mean = np.zeros((1, 3), dtype=float)
    std = np.zeros((1, 3), dtype=float)

    # calculate mean and std
    for img in image_list:
        mean += np.array([[np.mean(img[:, :, 0]),
                           np.mean(img[:, :, 1]),
                           np.mean(img[:, :, 2])]])

        std += np.array([[np.sqrt(np.mean(np.square(img[:, :, 0] -
                                          np.mean(img[:, :, 0])))),
                          np.sqrt(np.mean(np.square(img[:, :, 1] -
                                          np.mean(img[:, :, 1])))),
                          np.sqrt(np.mean(np.square(img[:, :, 2] -
                                          np.mean(img[:, :, 2]))))]])
    mean = mean / len(image_list)
    std = std / len(image_list)

    return mean, std


def separate_images(filenames):
    """Separate images with respect to darkness, blurriness and brightness.

    This function tries to separate images according to their visual
    properties, i.e. if they are dark, blurry or bright.
    args:
    - filenames [list[str]]: list of filenames
    returns:
    - dark_imgs [list]: list of filenames belonging to dark images
    - blurry_imgs [list]: list of filenames belonging to blurry images
    - bright_imgs [list]: list of filenames belonging to bright images
    """
    datasets = [get_dataset(filename) for filename in filenames]
    batch = [sample for data in datasets
             for sample in data.shuffle(1000).take(1)]

    image_dict = {correct_filename(sample['filename'].numpy().decode('utf-8')):
                  sample['image'].numpy() for sample in batch}

    # HLS images are used to distinguish dark from bright images
    # by the lightness channel
    hls_image_list = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HLS),
                              image_dict.values()))

    hls_mean, hls_std = calculate_mean_std(hls_image_list)

    dark_imgs = []
    blurry_imgs = []
    bright_imgs = []

    for fn, img in image_dict.items():
        # Get mean and std for hls image
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        mean, std = calculate_mean_std([hls_img])

        # Compute focus measure to detect blurry images
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if np.less(mean[..., 1], hls_mean[..., 1] - hls_std[..., 1]):
            dark_imgs.append(fn)
        elif fm < 350:
            blurry_imgs.append(fn)
        else:
            bright_imgs.append(fn)

    return dark_imgs, blurry_imgs, bright_imgs


def split(source, destination):
    """Create two splits from the processed records.

    The files should be moved to new folders in the same directory.
    These folders should be named train and val.

    args:
        - source [str]: source data directory, contains the
                        processed tf records
        - destination [str]: destination data directory, contains 2 sub
                             folders: train / val
    """
    # 80% of the data are taken for training
    train_ratio = 0.8

    source_path = source
    source = os.path.join(source, '*.tfrecord')
    train_path = os.path.join(destination, 'train')
    val_path = os.path.join(destination, 'val')

    filenames = []
    for filename in glob.iglob(source, recursive=True):
        filenames.append(filename)

    dark_imgs, blurry_imgs, bright_imgs = separate_images(filenames)
    num_dark = len(dark_imgs)
    num_blurry = len(blurry_imgs)
    num_bright = len(bright_imgs)

    train_idx_dark = random.sample(range(num_dark),
                                   int(num_dark * train_ratio))
    train_idx_blurry = random.sample(range(num_blurry),
                                     int(num_blurry * train_ratio))
    train_idx_bright = random.sample(range(num_bright),
                                     int(num_bright * train_ratio))

    for idx in range(num_dark):
        if idx in train_idx_dark:
            shutil.move(os.path.join(source_path, dark_imgs[idx]),
                        os.path.join(train_path, dark_imgs[idx]))
        else:
            shutil.move(os.path.join(source_path, dark_imgs[idx]),
                        os.path.join(val_path, dark_imgs[idx]))

    for idx in range(num_blurry):
        if idx in train_idx_blurry:
            shutil.move(os.path.join(source_path, blurry_imgs[idx]),
                        os.path.join(train_path, blurry_imgs[idx]))
        else:
            shutil.move(os.path.join(source_path, blurry_imgs[idx]),
                        os.path.join(val_path, blurry_imgs[idx]))

    for idx in range(num_bright):
        if idx in train_idx_bright:
            shutil.move(os.path.join(source_path, bright_imgs[idx]),
                        os.path.join(train_path, bright_imgs[idx]))
        else:
            shutil.move(os.path.join(source_path, bright_imgs[idx]),
                        os.path.join(val_path, bright_imgs[idx]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
