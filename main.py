import os
import sys

import cv2
import random
import argparse
import collections
import time
from tqdm import tqdm
import copy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor, ImageFont

import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import nengo_dl
import nengo
import pandas as pd

from utils import *



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=bool)
    parser.add_argument('--test', type=bool)

    parser.add_argument('--batch_size', type=int, default=1)


    args = parser.parse_args()
    return args


def

if __name__ == '__main__':
    