import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from app.lib.train import train


def main(_):
    train()

#Reponsável por treinar a LSTM. Caso a rede já tenha sido treinado o novo treinamento
# será incremental, partindo do ponto onde o ultimo treinamento parou.
if __name__ == "__main__":
    tf.app.run()