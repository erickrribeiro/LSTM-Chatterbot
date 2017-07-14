## ChatterBot para AMMD 

> *Importante : Este repositório foi desenvolvido como uma forma de treinar um rede neural LSTM, mas precisamente
uma rede sequence to sequence afim de treinar um bot de proposito geral, e adiquirir experiencia com Deep Learning.*

**Setup**
    
    git clone https://github.com/erickrribeiro/LSTM-Chatterbot.git
    cd LSTM-Chatterbot
   
Durante o decorrer das épocas no treinamento da rede neural este projeto fará uma serie de backups, para garantir que a 
 qualquer momento será possível parar o treinamente, avaliar o desempenho da rede, e continuar e onde parou caso seja
 necessário.
 
 Para isso é necessário executar o seguinte script:
 
    ./setup 
 Como resultado será criado o seguinte esquema de pastas, onde
 os metadados da rede serão armazenados.
    
```
└── experiment                    --  
    ├── data                      -- 
    ├── nn_models                 --  
    └── results                   -- 
```
The current results are pretty lousy:

    hello baby	            - hello
    how old are you ?           - twenty .
    i am lonely	            - i am not
    nice                        - you ' re not going to be okay .
    so rude	                    - i ' m sorry .
    

**Papers**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)

**Arquitetura**

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)
    
**Run**


    python train.py
    
Testar o chatbot pra um conjunto de frases pré-definidas:

    python test.py
    
Playground:

    python chat.py
    
Todos os parametros de configuração esão em `app/configs/config.py`


**Requirements**

* [tensorflow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
