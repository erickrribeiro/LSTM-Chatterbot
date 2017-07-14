#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import math
import time

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from app.lib.seq2seq_model_utils import create_model
from app.configs.config import FLAGS, BUCKETS
from app.lib.data_utils import read_data
from app.lib import data_utils

def train():
    print ("Preparando dados do dataset em %s"% FLAGS.data_dir)

    train_data, dev_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:

        #Criando o modelo, internamete verifica se já foi criado.
        print ("Criando %d camada(s) de tamanho %d . "%(FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, forward_only=False)

        #Lendo dados de treino e colocando nos buckets.
        print ("Carregando os dados da base de treino (limite: %d)."%(FLAGS.max_train_data_size))
        dev_set = read_data(dev_data)
        train_set = read_data(train_data, FLAGS.max_train_data_size)

        print ("Configurando tamanho do buckets no treino.")
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
          # É escolhido um bucket de acordo com a distribuição de dados. Escolhemos um número
          # aleatório entre [0, 1] e usamos o intervalo correspondente em train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([
              i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01
          ])

          #Obtem o batch e avança.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

          _, step_loss, _ = model.step(
              sess,
              encoder_inputs,
              decoder_inputs,
              target_weights,
              bucket_id,
              forward_only=False)

          #Calcula o tempo gasta para executar esse passo.
          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1

          # Salvando o chekpoint e imprimo estatísticas da epoca na tela.
          if current_step % FLAGS.steps_per_checkpoint == 0:
            # Imprimir estatísticas para a época anterior.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print("global step %d learning rate %.4f step-time %.2f perplexity %.2f"%(model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
 
            # Decrease learning rate if no improvement was seen over last 3 times.

            #print ("--------------------------")
            #print ("Numero de batches visto %4f"%(model.global_step.eval))
            #print ("Taxa de aprendizado %4f"%(model.learning_rate.eval()))
            #print ("Tempo gasto: %.2f"%(step_time))
            #print ("Perplexidade: %.2f"%(perplexity))
            #print ("--------------------------")

            # Diminua a taxa de aprendizado se nenhuma melhoria foi observada nas últimas 3 vezes.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # Salva o checkpoint e zera o timer e o loss.
            checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0

            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(BUCKETS)):
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
              _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

            sys.stdout.flush()
