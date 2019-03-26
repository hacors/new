import os
import tensorflow as tf

from utils import lstm, stack_lstm


class Encoder(tf.keras.Model):
    """
    Encoder module of Seq2Seq
    Extends from tf.keras.Model for eheckpointing and get trainable variables
    """

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 n_hidden,
                 type='lstm'):
        super(Encoder, self).__init__()
        self.n_hidden = n_hidden

        with tf.variable_scope("encoder"):
            self.w_emb = tf.get_variable(name="w_emb",
                                         shape=[vocab_size, n_hidden])
            self.w_lstm = tf.get_variable("w_lstm", [2 * n_hidden,
                                                     4 * n_hidden])

    def call(self, sent, hidden, is_train=True):
        """
        Using tf.while_loop to implement this function according to our lectures. 

        Args:
          sent: input (a whole source sentence) 
          hidden: 
          is_train: if you plan to use drop_out then make us of this  

        Returns:
          enc_outputs (all hiddens states) and last hidden state 
        """
        B, L = tf.shape(sent)
        hidden_states = tf.TensorArray(tf.float32, size=L, clear_after_read=False)

        # loop until the end of max sentence length of this batch
        def condition(i, sent, *args): pass

        def body(i, sent, prev_c, prev_h, hidden_states):
            pass
            return i + 1, sent, next_c, next_h, hidden_states

        # main loop of LSTM
        loop_vars = None
        loop_outputs = tf.while_loop(condition, body, loop_vars)
        hidden_states = None
        return hidden_states, hidden_states[:, -1, :]


class Decoder(tf.keras.Model):
    """
    Decoder module of Seq2Seq
    Extends from tf.keras.Model for eheckpointing and get trainable variables
    """

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 n_hidden,
                 type='lstm',
                 attention_type='bahdanau'):
        super(Decoder, self).__init__()
        self.n_hidden = n_hidden
        self.attention_type = attention_type

        with tf.variable_scope("decoder"):
            self.w_emb = tf.get_variable(name="w_emb", shape=[vocab_size, n_hidden])
            self.w_lstm = tf.get_variable("w_lstm", [2 * n_hidden, 4 * n_hidden])
            self.w_soft = tf.get_variable("w_soft", [n_hidden, vocab_size])

        with tf.variable_scope("attention"):
            # only used for attention
            # you can implement your own weights/shape
            self.V = tf.get_variable('v', [2*n_hidden, n_hidden])

    def call(self, sent, hidden, enc_output, is_attention=True):
        """

        Args:
          sent: a single token only => B x 1 to be converted to B x Emb  
          hidden: of previous step, the very first one is the encoder's output 
          enc_output: B x E x H - full history of encoder's hidden (out) states 
                      for calculation of attention 
          is_attention: or not  

        Returns:

        """
        attention_weights = None

        B, L = tf.shape(sent)  # redundant but accept for now, we only have L==1
        hidden_states = tf.TensorArray(tf.float32, size=L, clear_after_read=False)

        # loop until the end of max sentence length of this batch
        def condition(i, sent, *args): pass

        def body(i, sent, prev_c, prev_h, hidden_states):
            # only 1 time step
            # get embedding represetnation and infer through LSTM functino
            nextc, next_h = lstm(...)

            # trick for attention
            score = None
            if is_attention:
                # make use of enc_output
                next_h = next_h

            hidden_states = hidden_states.write(i, next_h)
            return i + 1, sent, next_c, next_h, hidden_states, score

        # main loop of LSTM, pay attention to initialization carefully!
        loop_vars = None
        loop_outputs = tf.while_loop(condition, body, loop_vars)
        hidden_states = None

        if is_attention:
            attention_weights = loop_outputs[-1]

        # expand to Vocabulary
        hidden_out = None
        output = None

        return output, hidden_out, attention_weights


def model(data_dict,
          checkpoint_dir='checkpoint',
          emb_dim=256,
          n_hidden=1024,
          type='gru'):
    """No need to change"""
    encoder = Encoder(data_dict['src_vocab_size'],
                      emb_dim,
                      n_hidden,
                      type)
    decoder = Decoder(data_dict['tgt_vocab_size'],
                      emb_dim,
                      n_hidden,
                      type,
                      attention_type='bahdanau')
    ckpt_prefix = os.path.join(checkpoint_dir, "ckpt")
    optimizer = tf.train.AdamOptimizer()

    # checkpoint will not only save the model but also the optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    return {
        'encoder': encoder,
        'decoder': decoder,
        'train_ckpt': checkpoint,
        'ckpt_prefix': ckpt_prefix,
        'optim': optimizer,
    }
