#!/usr/bin/python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function

import os
from six.moves import cPickle


import tensorflow as tf
from model import Model


default_context = ['Why do many people say Windows sucks?',
                   'What makes Linux such a great choice that it’s used in all those devices?',
                   'It’s because it’s open source software, which has various implications.'
                   ]
default_last_utterance = 'What can’t Linux do?'

class Bot(object):
    def load(self, model_dir):
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(model_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)

        self.model = Model(saved_args, training=False) # it takes 10^-1 seconds
        self.sess = sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables()) # it takes 10^-2 seconds
        ckpt = tf.train.get_checkpoint_state(model_dir) # it takes 10^-3 seconds

        assert ckpt and ckpt.model_checkpoint_path, 'Can not get checkpoint'
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def inference(self, kwargs: dict):
        contex = kwargs.get('contex', default_context)
        last_utterance = kwargs.get('last_utterance', default_last_utterance)
        utterance_option = kwargs.get('utterance_option', 0)

        contex += [last_utterance]
        contex = '\n'.join(contex)
        contex = ''.join(ch for ch in contex if ch in self.vocab)

        out_utterances = self.model.sample(self.sess, self.chars, self.vocab, 500, contex,
                        2)[len(contex):] # it takes 1 seconds
        out_utterances = [utterance for utterance in out_utterances.split('\n') if utterance]
        utterance_option = min(abs(utterance_option), len(out_utterances))
        out_utterance = out_utterances[utterance_option]
        return out_utterance


if __name__ == '__main__':
    bot = Bot()
    bot.load('save')
    out_utterance = bot.inference({'contex':['Peace bro'],'last_utterance':'What can’t Linux do?','utterance_option':0})
    print(out_utterance)
