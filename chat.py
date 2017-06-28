import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from app.lib.chat import chat


def main(_):
    chat()
#Reponsável por carregar a LSTM treinada e abrir uma interface de chat com usuário,
#onde o usuário pode interagir com o bot e receber respostas em tempo real.
if __name__ == "__main__":
    tf.app.run()