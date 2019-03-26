from __future__ import absolute_import, division, print_function

import os
import random
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()  # not needed if you upgrade to TF 2.0

from data_utils import BOS, EOS
from data_utils import load_dataset
from data_utils import create_batched_dataset
from data_utils import sentence_to_tensor
from models import model

from utils import DEFINE_boolean
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import DEFINE_float
from utils import print_user_flags
from utils import loss_function
from utils import plot_attention_map

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_boolean("restore_checkpoint", True, "Auto retrieve checkpoint or not.")
DEFINE_boolean("is_toy", False, "Toy Dataset or not, useful for debugging")
DEFINE_boolean("is_attention", False, "Using Attention for Decoder or not")
DEFINE_string("data_path", "en-es", "Path to parallel corpus data (.txt)")
DEFINE_string("output_dir", "output", "Path to log folder")
DEFINE_string("cell_type", "gru", "GRU or LSTM or naive, ...")
DEFINE_integer("n_epochs", 10, "Number of training epochs")
DEFINE_integer("n_layers", 1, "Number of stacked RNN layers")
DEFINE_integer("n_hidden", 1024, "Dimensionality of RNN output")
DEFINE_integer("emb_dim", 256, "Dimensionality of word embedding, src==tgt")
DEFINE_integer("save_every", 500, "How many batches to save once")
DEFINE_integer("eval_every", 100, "How many batches to evaluate")
DEFINE_integer("batch_size", 64, "Batch size. SET to `2` for easy debugging.")
DEFINE_integer("n_loaded_sentences", 20000, "Number of sentences to load, "
                                            "Set to <= 0 for loading all data,"
                                            "SET LOWER FOR DEBUGGING")
DEFINE_float("init_lr", 1e-3, "Init learning rate. This is default for Adam.")
DEFINE_float("drop_keep_prob", 1.0, "Dropout rate")


def train():
  """
  Train driver - no need to change.  
  """
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {0} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {0} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print_user_flags()

  # --------------------------------
  #             DATASET
  # --------------------------------
  data_dict = load_dataset(FLAGS.data_path,
                           is_toy=FLAGS.is_toy,
                           num_examples=FLAGS.n_loaded_sentences)
  batched_datasets = create_batched_dataset(data_dict,
                                            batch_size=FLAGS.batch_size)

  # --------------------------------
  #             MODEL
  # --------------------------------
  nmt_model = model(data_dict,
                    checkpoint_dir=FLAGS.output_dir,
                    emb_dim=FLAGS.emb_dim,
                    n_hidden=FLAGS.n_hidden,
                    type=FLAGS.cell_type)
  encoder, decoder = nmt_model['encoder'], nmt_model['decoder']

  # --------------------------------
  #             RESTORE
  # --------------------------------
  # try to restore saved model
  if FLAGS.restore_checkpoint:
    try:
      nmt_model['train_ckpt']. \
        restore(tf.train.latest_checkpoint(FLAGS.output_dir))
      print("\n\nTRYING TO RESTORE THE LATEST MODEL FROM: "
            "'{}'\n\n".format(FLAGS.output_dir))
    except Exception as e:
      print("Error in restoring checkpoint: {}. Training from "
            "scratch".format(e))

  # --------------------------------
  #             TRAIN
  # --------------------------------
  # (continue to) train now
  for epoch in range(1, FLAGS.n_epochs + 1):
    start = time.time()
    total_loss = 0
    total_ppl = 0

    # reset encoder's hidden state at the beginning of every epoch to ZEROs
    hidden = tf.zeros((FLAGS.batch_size, FLAGS.n_hidden))

    for (batch_num, (src, tgt)) in enumerate(batched_datasets['train']):
      loss = 0
      with tf.GradientTape() as tape:
        loss = teacher_forcing(data_dict,
                               encoder,
                               decoder,
                               hidden,
                               loss,
                               src,
                               tgt)# forward
      batch_loss = (loss / int(tgt.shape[1]))
      batch_ppl = np.exp(batch_loss)
      total_loss += batch_loss
      total_ppl += batch_ppl

      # backward
      variables = encoder.variables + decoder.variables
      gradients = tape.gradient(loss, variables)
      nmt_model['optim'].apply_gradients(zip(gradients, variables))

      if batch_num % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f} '
              'PPL {:.2f}'.format(epoch,
                                  batch_num,
                                  batch_loss.numpy(),
                                  batch_ppl))
      # evaluate some sentences
      if (batch_num + 1) % FLAGS.eval_every == 0:
        sentences = draw_random_sentences(os.path.join(FLAGS.data_path,
                                                       'val.txt'),
                                          size=2)
        for sentence in sentences:
          # change False to True for Attention Plots
          translate(sentence, encoder, decoder, data_dict, FLAGS.is_toy,
                    FLAGS.is_attention, False)

      # saving (checkpoint) the model every 2 epochs
      if (batch_num + 1) % FLAGS.save_every == 0:
        nmt_model['train_ckpt'].save(file_prefix=nmt_model['ckpt_prefix'])
        print("Model saved to '{}".format(FLAGS.output_dir))

    print('Epoch {} Loss {:.4f} PPL {:.2f}'.format(epoch,
                                                   total_loss / batch_num + 1,
                                                   total_ppl / batch_num + 1))
    print("Saving for this epoch")
    nmt_model['train_ckpt'].save(file_prefix=nmt_model['ckpt_prefix'])

    print('Time taken for this epoch {:.2f} sec\n'.format(time.time() - start))
    print("-" * 80)
    sys.stdout.flush()

    # final test after training
    # you can inject this into the loop to do it at every epoch's end
    if epoch == FLAGS.n_epochs:
      print("Testing the whole test set for epoch {}".format(epoch))
      test_lines = open(os.path.join(FLAGS.data_path, 'target.txt')).\
        read().strip().split('\n')
      with open(os.path.join(FLAGS.data_path, 'translated_{}.txt'.format(epoch)),
                'w') as f:
        for line in test_lines:
          result, _, _ = evaluate(line, encoder, decoder, data_dict,
                                  FLAGS.is_toy, FLAGS.is_attention)
          f.write(' '.join(result.split()[:-1]) + '\n')


def teacher_forcing(data_dict, encoder, decoder, hidden, loss, src, tgt):
  """
  Training using Teacher-Forcing mode, accumulate loss function at every 
  decoding timestep using loss_function().  
  
  Remember the very first size of `src` and `tgt` is FLAGS.batch_size  
  
  Args:
    data_dict: for converting WORDS to INDEXES, needing for pre-padding BOS 
                at the decoding time step 0 
    encoder: 
    decoder: 
    hidden: hidden input here should be all Zeros 
    loss: input here should be all zeros, then get accumulated at EVERY 
          timestep 
    src: batch of source sentences 
    tgt: batch of target sentences 

  Returns:
    loss value accumulated (not averaged yet) 

  """
  # TODOS: infer through encoder


  # Decoder: Start with a batch of BOS tokens
  BOS_idx = data_dict['src_lang'].word2idx[BOS.strip()]
  dec_input = tf.expand_dims([BOS_idx] * FLAGS.batch_size, 1)

  # At every timestep, feed GROUND TRUTH not predicted token
  for t in range(1, tgt.shape[1]):
    # for each step, ig nore the output of prediction an use teacher's input
    # accumulate the loss

  return loss


def draw_random_sentences(data_path, size=2):
  """ Draw random sentences from val.txt for evaluation"""
  lines = open(data_path).read().splitlines()

  sentences = []
  for _ in range(size):
    sentences.append(random.choice(lines).strip().split('\t')[-1])

  return sentences


def evaluate(sentence,
             encoder,
             decoder,
             data_dict,
             is_toy,
             is_attention=False):
  """
  
  Unlike teacher forcing which consumes ground truth at every step, in this 
  driver you need to use the predicted token from the previous step to feed 
  to the current one. 
  
  Normally you can do it step by step until the end, beginning from 
  
  Args:
    sentence: input to translate, from 'target.txt' without pre-processing 
    encoder: 
    decoder: 
    data_dict: 
    is_toy: 
    is_attention: 

  Returns:
    result: decoded sentence in language (not numbers) 
    sentence: untouched input sentence 
    attention_plots: 2D matrix of attention plot 

  """
  attention_plot = None
  if is_attention:
    attention_plot = np.zeros((data_dict['max_length_tgt'],
                               data_dict['max_length_src']))
  # convert sentence to tensor
  inputs_tensor = sentence_to_tensor(data_dict, is_toy, sentence)

  # encode
  hidden = [tf.zeros((1, FLAGS.n_hidden))]


  # TODOs
  # 1. Inference through encoder with init hidden state above
  # 2. Use encoder's state and output to input to Decoder. Important to note
  # that you should starts with BOS at the beginning of decoding.
  # and follow the guidance below:


  # infer here and init Decoder

  # very first step to zeros for the first step. Only 1 sentence in this batch.
  BOS_idx = data_dict['tgt_lang'].word2idx[BOS.strip()]
  dec_input = tf.expand_dims([BOS_idx], 0)

  result = ''
  for t in range(data_dict['max_length_tgt']):
    # decode each step using decoder

    # if attention then update attention plot

    # important: if you see the EOS toke, then the decoding is done
    # otherwise you need to do it to maximally end of this loop


    # accumulate the result as well
    # each step should be a word decoded from your idx2word dict in data_dict
    # of the concerning target language
    predictions, dec_hiddens, attention_weights = decoder(...)

    # storing the attention weights to plot later on
    if is_attention:
      attention_weights = tf.reshape(attention_weights, (-1,))
      attention_plot[t] = attention_weights.numpy()


    # get the id of the predicted token and figure out the corresponding word
    # in the respective dictionary

    # some more hints in real code - you don't have to use

    # end if predicts the EOS token
    if data_dict['tgt_lang'].idx2word[predicted_id] == EOS.strip():
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


def translate(sentence,
              encoder,
              decoder,
              data_dict,
              is_toy=False,
              is_attention=False,
              is_plot=False):
  """Translate with input of a sentence (in words, not in numbers) and the 
  translated output with the same format. No need to change"""
  result, sentence, attention_plot = evaluate(sentence,
                                              encoder,
                                              decoder,
                                              data_dict,
                                              is_toy,
                                              is_attention)

  print('\tInput:\t\t{}'.format(sentence))
  print('\tPredicted:\t\t{}'.format(result))

  if is_attention:
    attention_plot = attention_plot[:len(result.split(' ')),
                     :len(sentence.split(' '))]
    if is_plot:
      plot_attention_map(attention_plot, sentence.split(' '), result.split(' '))



if __name__ == "__main__":
  train()
