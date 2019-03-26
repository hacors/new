# -*- coding: utf-8 -*-

from collections import defaultdict
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import unicodedata

import tensorflow as tf
tf.enable_eager_execution()  # not needed if you upgrade to TF 2.0


# constants
BOS = "<s> "
EOS = " </s>"
UNK = "<unk>"
PAD = "<pad>"


def preprocess_sentence(w, is_toy=False):
    # Converts the unicode file to ascii
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    w = unicode_to_ascii(w.lower().strip())

    if not is_toy:
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # padding a start and an end token to the sentence
    w = BOS + w + EOS
    return w


def create_dataset(path, num_examples, is_toy=False):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    all_lines = lines if num_examples <= 0 else lines[:num_examples]
    word_pairs = [[preprocess_sentence(w, is_toy) for w in l.split('\t')]
                  for l in all_lines]
    return word_pairs


class WordAndIdx(object):
    """
    word2idx and idx2word  
    """

    def __init__(self, lang):
        self.UNK_TOKEN_IDX = 1
        self.lang = lang
        self.word2idx = defaultdict(lambda: self.UNK_TOKEN_IDX)
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx[PAD] = 0
        self.word2idx[UNK] = self.UNK_TOKEN_IDX
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 2

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def sentence_to_tensor(data_dict, is_toy, sentence):
    # preprocess input sentence
    sentence = preprocess_sentence(sentence, is_toy)
    inputs = [data_dict['src_lang'].word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence. \
        pad_sequences([inputs],
                      maxlen=data_dict['max_length_src'],
                      padding='post')
    inputs = tf.convert_to_tensor(inputs)
    return inputs


def load_dataset(dir_path, num_examples, test_pct=0.2, is_toy=False):
    def load_single_dataset(path):
        # creating cleaned input, output pairs
        pairs = create_dataset(path, num_examples, is_toy)

        # index language using the class defined above
        src_lang = WordAndIdx(t for s, t in pairs)
        tgt_lang = WordAndIdx(s for s, t in pairs)

        # source language sentences
        source_tensor = [[src_lang.word2idx[t] for t in tgt.split(' ')]
                         for src, tgt in pairs]

        # target language sentences
        target_tensor = [[tgt_lang.word2idx[s] for s in src.split(' ')]
                         for src, tgt in pairs]

        # Calculate max_length of input and output tensor
        # Here, we'll set those to the longest sentence in the dataset
        max_length_src, max_length_tgt = \
            max_length(source_tensor), max_length(target_tensor)

        # Padding the input and output tensor to the maximum length
        source_tensor = tf.keras.preprocessing.sequence. \
            pad_sequences(source_tensor,
                          maxlen=max_length_src,
                          padding='post')

        target_tensor = tf.keras.preprocessing.sequence. \
            pad_sequences(target_tensor,
                          maxlen=max_length_tgt,
                          padding='post')

        return source_tensor, target_tensor, src_lang, tgt_lang, \
            max_length_src, max_length_tgt

    # training dataset
    src_train, tgt_train, src_lang, tgt_lang, max_length_src, max_length_tgt = \
        load_single_dataset(os.path.join(dir_path, 'train.txt'))

    # val/test dataset
    src_val, tgt_val, _, _, _, _ = \
        load_single_dataset(os.path.join(dir_path, 'val.txt'))

    data_dict = {
        "src_train": src_train,
        "src_val": src_val,
        "tgt_train": tgt_train,
        "tgt_val": tgt_val,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_vocab_size": len(src_lang.word2idx),
        "tgt_vocab_size": len(tgt_lang.word2idx),
        "max_length_src": max_length_src,
        "max_length_tgt": max_length_tgt
    }

    return data_dict


def create_batched_dataset(data_dict,
                           batch_size=32,
                           buffer_size=10000):
    train_dataset = tf.data.Dataset.from_tensor_slices((data_dict['src_train'],
                                                        data_dict['tgt_train']))
    train_dataset = train_dataset.shuffle(buffer_size=len(data_dict["src_train"]))
    batched_train_dataset = train_dataset.batch(batch_size=batch_size,
                                                drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((data_dict['src_val'],
                                                      data_dict['tgt_val']))
    batched_val_dataset = val_dataset.batch(batch_size=batch_size,
                                            drop_remainder=False)

    return {
        "train": batched_train_dataset,
        "val": batched_val_dataset
    }


if __name__ == "__main__":
    # no need to change anything in this file, make sure this test can run
    # small test driver
    # pass all of those below before moving on
    data_path = 'en-fr'
    data_dict = load_dataset(data_path, num_examples=200)

    print(len(data_dict['src_train']))
    print(len(data_dict['tgt_train']))
    print(len(data_dict['src_val']))
    print(len(data_dict['tgt_val']))

    dataset_dict = create_batched_dataset(data_dict, batch_size=10)

    c = 0  # number of batches
    for batch_data in dataset_dict['train']:
        c += 1
        print(type(batch_data), batch_data)

    print(c)

    # import pdb; pdb.set_trace()
