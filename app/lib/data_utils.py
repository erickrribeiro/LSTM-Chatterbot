#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
from app.configs.config import BUCKETS

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Expressões regulares utilizadas no tokenizer
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
#Utilizado para verificar a existencia de uma dígito.
_DIGIT_RE = re.compile(r"\d{3,}")

def get_dialog_train_set_path(path):
  return os.path.join(path, 'chat')


def get_dialog_dev_set_path(path):
  return os.path.join(path, 'chat_test')


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """
   Criar arquivo de vocabulário (se ainda não existir).

   O arquivo de vocabulário foi projetado para conter uma palavra por linha. Cada sentença é
   Tokenized e os dígitos são normalizados (se normalize_digits estiver configurado).
   O vocabulário contém os tokens mais freqüentes até o max_vocabulary_size.
   O arquivo de vocabulários em um formato de um token por linhas, para que facilitar o acesso,
   pois o token na primeira linha terá id = 0, a segunda linha id = 1, e assim por diante.

  :param vocabulary_path: String Caminho onde o arquivo de vocabulários será criado
  :param data_path: String Caminho de onde os vocabulários viram.
  :param max_vocabulary_size: Integer limite máximo de linhas do arquivo de vocabulários
  :param tokenizer: Função Função para tokenizar cada frase, Caso seja None será utilizado
  basic_tokenizer.
  :param normalize_digits: booleano Caso true, todos os dígitos serão substituídos por 0s.
  :return:
  """
  if not gfile.Exists(vocabulary_path):
    print ("Criando arquivo de vocabulário %s a partir dos dados de %s"%(vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as file:
      counter = 0
      for line in file:
        counter += 1
        if counter % 100000 == 0:
          print ("Processando linha %d"%(counter))
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w

          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

      #remove a quantidade excedente de vocabulários
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      #cria o arquivo de vocabulários com a quantidade máxima de vocabulários.
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []

    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())

    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab

  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """
  Converta uma sequência de palavras para a lista de inteiros que chamadas de token-ids.

   Por exemplo, uma frase "I have a dog" pode ser tokenizada como
   ["I", "have", "a", "dog"] e com vocabulário {"I": 1, "have": 2,
   "a": 4, "dog": 7 "} esta função irá retornar [1, 2, 4, 7].

  :param sentence: String Frase para converter em token-ids.
  :param vocabulary: Dicionario Responsável por mapear tokens em inteiros:
  :param tokenizer: Função, Função utilizada para tokenizar cada sentença, caso
  a função não seja passada a função basic_tokenizer será utilizado;
  :param normalize_digits: Boolean Caso True, todos os dígitos são substituídos por 0s
  :return: Uma lista de inteiros.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(word, UNK_ID) for word in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", word), UNK_ID) for word in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_dialog_data(data_dir, vocabulary_size):
  """Obter dados de diálogo em data_dir, criar vocabulários e tokenizar dados.
  :param data_dir: diretório no qual os conjuntos de dados serão armazenados.
  :param vocabulary_size: tamanho do vocabulário em inglês para criar e usar.
  :return:
       Uma tupla de 3 elementos:
       (1) caminho para o token-ids para o conjunto de dados de treinamento,
       (2) caminho para o token-ids para o conjunto de dados de desenvolvimento do bate-papo,
       (3) caminho para o arquivo de vocabulário de bate-papo

  """
  # Get dialog data to the specified directory.
  train_path = get_dialog_train_set_path(data_dir)
  dev_path = get_dialog_dev_set_path(data_dir)

  #Criando arquivo de vocabulários (caso não exista) de acordo com a quantidade máxima de palavras.
  vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
  create_vocabulary(vocab_path, train_path + ".in", vocabulary_size)

  # Create token ids for the training data.
  train_ids_path = train_path + (".ids%d.in" % vocabulary_size)
  data_to_token_ids(train_path + ".in", train_ids_path, vocab_path)

  # Create token ids for the development data.
  dev_ids_path = dev_path + (".ids%d.in" % vocabulary_size)
  data_to_token_ids(dev_path + ".in", dev_ids_path, vocab_path)

  return (train_ids_path, dev_ids_path, vocab_path)


def read_data(tokenized_dialog_path, max_size=None):
  """Read data from source file and put into buckets.

  Args:
    source_path: path to the files with token-ids.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in BUCKETS]

  with gfile.GFile(tokenized_dialog_path, mode="r") as file:
      source, target = file.readline(), file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  Lendo as %d primeiras linhas"%(counter))
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = file.readline(), file.readline()
  return data_set