#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from app.configs.config import TEST_DATASET_PATH, FLAGS
from app.lib import data_utils
from app.lib.seq2seq_model_utils import load_model, get_predicted_sentence

def read_dataset(path):
    """
    Responsável por ler um dataset com base no caminho passado.
    :param path: caminho onde o dataset está localizado.

    :return: array de sentenças do dataset.
    """
    with open(path) as file:
        sentences = [s.strip() for s in file.readlines()]
    return sentences

def predict(checkpoint=None):
    if checkpoint==None:
        filename = '_'.join(['results', 'checkpoint', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    else:
        filename = '_'.join(['results', checkpoint, str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])

    path = os.path.join(FLAGS.results_dir, filename)

    with tf.Session() as session, open(path, 'w') as file:
        #Criando o modelo e carregando os paramentros.
        #model = create_model(session, forward_only=True)
        model = load_model(session,forward_only=True, checkpoint=checkpoint)
        model.batch_size = 1  # We decode one sentence at a time.

        print model
        #Carregando vocabularios
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        dataset = read_dataset(TEST_DATASET_PATH)

        for sentence in dataset:
            # Get token-ids for the input sentence.
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, session)
            print(sentence, ' -> ', predicted_sentence)
            file.write("%s -> %s\n"%(sentence,predicted_sentence))