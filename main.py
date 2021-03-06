import os
import sys
import re

import argparse
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import *
from model import Encoder, Decoder, Attention
from bleu import compute_BLEU

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--translate', default=False, action='store_true')

    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
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
        args.limit_size,
        reverse=True
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

    # split dataset into training and validation
    train_input, val_input, train_target, val_target = train_test_split(
        tensor_input,
        tensor_target,
        test_size = args.dev_split
    )

    # initialize batch size and consequent steps_per_epoch
    batch_size = args.batch_size
    steps_per_epoch = len(train_input) // batch_size

    # create tensorflow dataset by generating slices from the 
    # generated dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_input,train_target)).shuffle(len(train_input))
    dataset = dataset.batch(args.batch_size)

    # initialize SGD as the optimizer
    optimizer = tf.optimizers.SGD(args.learn_rate)

    # instantiate Encoder
    encoder = Encoder(
        len(tokenizer_input.word_index) + 1,
        args.embed_dims,
        args.enc_units,
        args.batch_size,
        args.dropout
        )

    # instantiate Decoder
    decoder = Decoder(
        len(tokenizer_target.word_index) + 1,
        args.embed_dims,
        args.enc_units,
        'concat',
        args.batch_size,
        args.dropout
        )
    
    # use Sparse Categorical Crossentropy as loss object
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    @tf.function
    def train_step(input, target, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            #pass input into the encoder
            enc_output, enc_hidden = encoder(input,enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [tokenizer_target.word_index['<s>']] * args.batch_size, 
                1
                )
            
            h_t = tf.zeros((args.batch_size, 1, args.embed_dims))

            for i in range(1, target.shape[1]):
                # passing enc_output into the decoder
                pred, dec_hidden, h_t = decoder(dec_input, dec_hidden, enc_output, h_t)

                loss += loss_function(target[:,i], pred, loss_object)

                # teacher forcing - passing the next target word into the decoder
                dec_input = tf.expand_dims(target[:,i],1)                                                                    

        # calculate the gradients and backpropogate via the optimizer
        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss
    
    # Initialize checkpoint saving
    checkpoint_prefix = os.path.join('checkpoints',args.out)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
        )

    # Run training for specified number of epochs
    for epoch in range(args.epochs):
        total_loss = 0

        # Initialize initial tensors
        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        enc_state = [[enc_hidden, enc_cell], [enc_hidden, enc_cell]]

        print('Starting Epoch',epoch+1)
        for(batch, (input,target)) in enumerate(dataset.take(steps_per_epoch)):
            # Run a train step for each batch in each epoch
            batch_loss = train_step(input, target, enc_state)
            total_loss += batch_loss

            # Print training progress
            if batch % 10 == 0:
                print('Epoch {}/{} Batch {}/{} Loss {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    batch + 10,
                    steps_per_epoch,
                    batch_loss.numpy()
                )
                )
        
        save_path = checkpoint.save(file_prefix=checkpoint_prefix)

# def translate(input_, lang_input, lang_target):
#     x = re.sub(r"([?.!,¿])", r" \1 ", input_)
#     x = re.sub(r'[" "]+', " ", x)
#     x = re.sub(r"[^a-zA-Z?.!,¿]+", " ", x)
#     processed = x.strip()
    
#     if args.reverse:
#         x_list = processed.split()
#         x_list.reverse()
#         processed = ' '.join(x_list)

#     processed = '<s> ' + processed + '</s>'

#     inputs = [lang_input.word_index[i] for i in sentence.split(' ')]
#     inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                             maxlen=args.max_len,
#                                                             padding='post')

#     inputs = tf.convert_to_tensor(inputs)

#     result = ''

#     hidden = [tf.zeros((1, units))]
#     enc_out, enc_hidden = encoder

if __name__ == '__main__':
    args = parse_args()
    tensor_input, tensor_target, tokenizer_input, tokenizer_target = data_gen()
    
    if args.train:
        train(tensor_input, tensor_target, tokenizer_input, tokenizer_target)
        
    if args.translate:
        input_sentence = input('Input sentence for translation (to Vietnamese): ')
        output_sentence = translate(input_sentence, tokenizer_input, tokenizer_target)
        print('Output: ' + output_sentence)
        print('')


