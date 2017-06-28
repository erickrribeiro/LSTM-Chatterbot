#!/usr/bin/env bash

# Criado diretórios os resultados serão armazanados.
save_dir='./experiment'
sudo mkdir -p $save_dir'/data/'
sudo mkdir -p $save_dir'/nn_models/'
sudo mkdir -p $save_dir'/results/'
sudo chown -R "$USER" $save_dir

# copy train and test data with proper naming
data_dir='app/data/train'
cp $data_dir'/movie_lines_selected.txt' $save_dir'/data/chat.in'
cp $data_dir'/movie_lines_selected_10k.txt' $save_dir'/data/chat_test.in'