## ChatterBot para AMMD 

> *Importante : Este repositório foi desenvolvido como uma forma de treinar um rede neural LSTM, mas precisamente
uma rede sequence to sequence afim de treinar um bot de proposito geral, e adiquirir experiencia com Deep Learning.*

Build a general-purpose conversational chatbot based on a hot 
seq2seq approach implemented in [tensorflow](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence_basics).
Since it doesn't produce good results so far, also consider other implementations of [seq2seq](https://github.com/nicolas-ivanov/seq2seq_chatbot_links).

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

    hello baby	        - hello
    how old are you ?   - twenty .
    i am lonely	        - i am not
    nice                - you ' re not going to be okay .
    so rude	            - i ' m sorry .
    
Disclaimer: 

* the answers are hand-picked (it looks cooler that way)
* chatbot has no power to follow the conversation line so far; in the example above it's a just a coincidence (hand-picked one)

Everyone is welcome to investigate the code and suggest the improvements.

**Actual deeds**

* realise how to diversify chatbot answers (currently the most probable one is picked and it's dull)


**Papers**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)

**Nice picture**

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)

Curtesy of [this](http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html) article.
    
**Run**

Train a seq2seq model on a small (17 MB) corpus of movie subtitles:

    python train.py
    
(this command will run the training on a CPU... GPU instructions are coming)

Test trained trained model on a set of common questions:

    python test.py
    
Chat with trained model in console:

    python chat.py
    
All configuration params are stored at `tf_seq2seq_chatbot/configs/config.py`

**GPU usage**

If you are lucky to have a proper gpu configuration for tensorflow already, this should do the job:

    python train.py
    
Otherwise you may need to build tensorflow from source and run the code as follows:

    cd tensorflow  # cd to the tensorflow source folder
    cp -r ~/tf_seq2seq_chatbot ./  # copy project's code to tensorflow root
    bazel build -c opt --config=cuda tf_seq2seq_chatbot:train  # build with gpu-enable option
    ./bazel-bin/tf_seq2seq_chatbot/train  # run the built code

**Requirements**

* [tensorflow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
