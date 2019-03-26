import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

user_flags = []


def DEFINE_string(name, default_value, doc_string):
  tf.app.flags.DEFINE_string(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
  tf.app.flags.DEFINE_integer(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
  tf.app.flags.DEFINE_float(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
  tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def print_user_flags(line_limit=80):
  print("-" * 80)

  global user_flags
  FLAGS = tf.app.flags.FLAGS

  for flag_name in sorted(user_flags):
    value = "{}".format(getattr(FLAGS, flag_name))
    log_string = flag_name
    log_string += "." * (line_limit - len(flag_name) - len(value))
    log_string += value
    print(log_string)


def leaky_relu(x, alpha=0.01):
  return tf.maximum(alpha * x, x)


def loss_function(real, pred):
  """Loss with mask to deal with variable length inputs"""
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real,
                                                         logits=pred) \
          * mask
  return tf.reduce_mean(loss_)


def count_model_params(tf_variables):
  """Count the total number of parameters in a model.

  Args:
    tf_variables: list of all model variables.
  """
  total = 0
  for v in tf_variables:
    shape = v.get_shape()
    variable_parameters = 1
    for dim in shape:
      variable_parameters *= dim.value
    total += variable_parameters
  print("Total number of parameters: " + str(total))
  return 0


def lstm(x, prev_c, prev_h, w):
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer,
                         trainable=trainable)


def create_bias(name, shape, initializer=None):
  if initializer is None:
    initializer = tf.constant_initializer(0.0, dtype=tf.float32)
  return tf.get_variable(name, shape, initializer=initializer)


def plot_attention_map(attention, input_sentence, predicted_sentence):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}
  ax.set_xticklabels([''] + input_sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  plt.show()