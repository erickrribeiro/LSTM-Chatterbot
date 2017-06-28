import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from app.lib.predict import predict


def main(_):
    predict()
#Reponsável por carregar a LSTM treinada e abrire e passar um conjunto de perguntas
# prontas a fim de medir a precisão do rede.
if __name__ == "__main__":
    tf.app.run()
