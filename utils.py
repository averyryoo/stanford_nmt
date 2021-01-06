import tensorflow as tf

import matplotlib.pyplot as plt
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
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    dataset = []
    for line in tqdm(lines[:limit_size]):
      sentence = line
      if reverse:
        line_list = line.split()
        line_list.reverse()
        sentence = ' '.join(line_list)
      
      processed_sentence = '<s> ' + sentence + ' </s>'
      dataset.append(processed_sentence)

    # lines = ['<s> ' + line + ' </s>' for line in lines[:limit_size]]

    return dataset

def tokenize(dataset, vocab, max_len=100):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters = '',
      oov_token = '<unk>'
  )

  tokenizer.word_index = vocab

  tensor = tokenizer.texts_to_sequences(dataset)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(
      tensor,
      maxlen=max_len,
      padding='post'
  )

  return tensor, tokenizer

def load_vocab(path, lang):
  lines = io.open(os.path.join(path,'vocab.{}'.format(lang)),encoding='UTF-8').read().strip().split('\n')
  vocab = {}

  for i, word in enumerate(lines):
    vocab[word] = i + 1
  
  return vocab

def load_dataset(path, lang, max_len=100, limit_size=None, reverse=False):
  text_path = os.path.join(path,'train.{}'.format(lang))
  vocab_path = os.path.join(path, 'vocab.{}'.format(lang))

  text = create_dataset(text_path,limit_size,reverse)
  vocab = load_vocab(path,lang)

  tensor, tokenizer = tokenize(text,vocab,max_len)

  return tensor, tokenizer

def loss_function(real,pred,object):
  mask = tf.math.logical_not(tf.math.equal(real,0))
  loss = object(real,pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  return tf.reduce_mean(loss)