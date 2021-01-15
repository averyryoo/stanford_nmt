import tensorflow as tf

import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
from tqdm import tqdm

def create_dataset(path, limit_size=None, reverse=False):
    # read file as list of lines
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    dataset = []
    for line in tqdm(lines[:limit_size]):
      sentence = line

      # Option to reverse the input sentence for better training results
      if reverse:
        line_list = line.split()
        line_list.reverse()
        sentence = ' '.join(line_list)
      
      # Add start and stop tokens
      processed_sentence = '<s> ' + sentence + ' </s>'
      dataset.append(processed_sentence)

    return dataset

def tokenize(dataset, vocab, max_len=100):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters = '',
      oov_token = '<unk>'
  )

  # initialize vocab dataset as .word_index
  tokenizer.word_index = vocab

  # transform text to sequence of integers
  tensor = tokenizer.texts_to_sequences(dataset)

  # pad sentences with 0s at the end according to max_len 
  tensor = tf.keras.preprocessing.sequence.pad_sequences(
      tensor,
      maxlen=max_len,
      padding='post'
  )

  # return the padded sequences and the tokenizer
  return tensor, tokenizer

def load_vocab(path, lang):
  # write dict of vocab words with ints as values
  lines = io.open(os.path.join(path,'vocab.{}'.format(lang)),encoding='UTF-8').read().strip().split('\n')
  vocab = {}

  for i, word in enumerate(lines):
    vocab[word] = i + 1
  
  return vocab

def load_dataset(path, lang, max_len=100, limit_size=None, reverse=False):
  text_path = os.path.join(path,'train.{}'.format(lang))
  vocab_path = os.path.join(path, 'vocab.{}'.format(lang))

  # load corpus and vocab
  text = create_dataset(text_path,limit_size,reverse)
  vocab = load_vocab(path,lang)

  tensor, tokenizer = tokenize(text,vocab,max_len)

  return tensor, tokenizer

def loss_function(real,pred,loss_object):
  mask = tf.math.logical_not(tf.math.equal(real,0))
  loss = loss_object(real,pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  return tf.reduce_mean(loss)