#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
from app.lib.train import train
from app.lib.test import tester
from app.lib.chat import chat

def start_train(_):
    train()

def start_test(_):
    checkpoint=None

    if len(sys.argv) == 3:
        checkpoint = sys.argv[2]

    tester(checkpoint=checkpoint)

def start_chat(_):
    chat()

def not_found_args():
    print ("Parâmetro obrigatório")
    print ("python start.py chat - Para iniciar uma conversa livre com o bot.")
    print ("python start.py test - Para testar o bot com perguntas prontas.")
    print ("python start.py train - Para treinar o bot.")
    exit(0)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        not_found_args()
    if sys.argv[1] == 'chat':
        tf.app.run(main=start_chat)
    elif sys.argv[1] == 'test':
        tf.app.run(main=start_test)
    elif sys.argv[1] == 'train':
        tf.app.run(main=start_train)
    else:
        not_found_args()

