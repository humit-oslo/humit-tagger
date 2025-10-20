#!/bin/bash

mkdir models/sentence_segmentation
mkdir models/classification
cd models/classification
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xaa
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xab
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xac
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xad
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xae
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xaf
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/xag
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/classification/config.json
cd ..
cd sentence_segmentation
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xaa
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xab
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xac
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xad
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xae
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xaf 
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/xag
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/config.json
wget https://github.com/humit-oslo/humit-tagger/raw/b3b88c6a3ebfe704f85b0f75a9a04c1f1eda0daa/models/sentence_segmentation/vocab.txt
cd ..
cd ..

ls models/*/x* &>/dev/null

if [[ "$?" -eq 0 ]]; then
    cat models/sentence_segmentation/x* > models/sentence_segmentation/pytorch_model.bin
    cat models/classification/x* > models/classification/pytorch_model.bin
    cat models/tokenization/x* > models/tokenization/pytorch_model.bin
    #rm models/*/x*
fi

pip3 install -r requirements.txt
