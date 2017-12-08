#!/usr/bin/env bash

cd ./data/word2vec/ &&
wget https://www.dropbox.com/s/t50fs4f74l4zbga/model.tar.gz && \
tar xvzf model.tar.gz && rm model.tar.gz
