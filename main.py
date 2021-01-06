import os
import sys

import argparse
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import *
from model import Encoder, Decoder, Attention

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--ckpt_file', type=str, default='checkpoints/ckpt_1.h5')
    parser.add_argument('--out', type=str, default='weights')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_lang', type=str, default='en')
    parser.add_argument('--target_lang', type=str, default='vi')
    parser.add_argument('--max_len', type=int, default=90)
    parser.add_argument('--limit_size', type=int)
    parser.add_argument('--dev_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=12)

    parser.add_argument('--embed_dims', type=int, default=500)
    parser.add_argument('--enc_units', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learn_rate', type=float, default=1.0)

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

    batch_size = args.batch_size
    steps_per_epoch = len(train_input) // batch_size

    dataset = tf.data.Dataset.from_tensor_slices((train_input,train_target)).shuffle(len(train_input))
    dataset = dataset.batch(args.batch_size)

    optimizer = tf.optimizers.SGD(args.learn_rate)

    encoder = Encoder(
        len(tokenizer_input.word_index) + 1,
        args.embed_dims,
        args.enc_units,
        args.batch_size,
        args.dropout
        )

    decoder = Decoder(
        len(tokenizer_target.word_index) + 1,
        args.embed_dims,
        args.enc_units,
        'concat',
        args.batch_size,
        args.dropout
        )
    
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    @tf.function
    def train_step(input, target, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(input,enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [tokenizer_target.word_index['<s>']] * args.batch_size, 
                1
                )
            
            h_t = tf.zeros((args.batch_size, 1, args.embed_dims))

            for i in range(1, target.shape[1]):
                pred, dec_hidden, h_t = decoder(dec_input, dec_hidden, enc_output, h_t)

                loss += loss_function(target[:,i], pred, loss_object)

                dec_input = tf.expand_dims(target[:,i],1)                                                                    

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss
    
    checkpoint_prefix = os.path.join('checkpoints',args.out)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
        )

    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        enc_state = [[enc_hidden, enc_cell], [enc_hidden, enc_cell]]

        print('Starting Epoch',epoch+1)
        for(batch, (input,target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(input, target, enc_state)
            total_loss += batch_loss

            if batch % 10 == 0:
                print('Epoch {}/{} Batch {}/{} Loss {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    batch + 10,
                    steps_per_epoch,
                    batch_loss.numpy()
                )
                )
        

        


if __name__ == '__main__':
    args = parse_args()
    tensor_input, tensor_target, tokenizer_input, tokenizer_target = data_gen()
    if args.train:
        train(tensor_input, tensor_target, tokenizer_input, tokenizer_target)
