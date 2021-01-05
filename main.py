import os
import sys

import random
import argparse
import collections
import time
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd

from utils import *
from model import Encoder, Decoder, Attention

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_lang', type=str, default='en')
    parser.add_argument('--target_lang', type=str, default='vi')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--limit_size', type=int)
    parser.add_argument('--dev_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=float, default=12)

    parser.add_argument('--embed_dims', type=float, default=500)
    parser.add_argument('--enc_units', type=float, default=)

    args = parser.parse_args()
    return args

def data_gen():
    print('Loading input dataset')
    tensor_input, tokenizer_input = load_dataset(
        args.data_dir,
        args.input_lang,
        args.max_len,
        args.limit_size
        )
    
    print('Loading target dataset')
    tensor_target, tokenizer_target = load_dataset(
        args.data_dir,
        args.target_lang,
        args.max_len,
        args.limit_size
        )
    
    print('Dataset loading complete')
    return tensor_input, tensor_target, tokenizer_input, tokenizer_target

def train(tensor_input, tensor_target, tokenizer_input, tokenizer_target):
    train_input, val_input, train_target, val_target = train_test_split(
        tensor_input,
        tensor_target,
        test_size = args.dev_split
    )

    dataset = tf.data.Dataset.from_tensor_slices((train_input,train_target))
    dataset = dataset.shuffle(len(train_input))

    print(dataset)

if __name__ == '__main__':
    args = parse_args()
    tensor_input, tensor_target, tokenizer_input, tokenizer_target = data_gen()
    if args.train:
        train(tensor_input, tensor_target, tokenizer_input, tokenizer_target)