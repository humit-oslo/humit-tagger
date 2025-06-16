#!/bin/bash

ls models/*/x* &>/dev/null

if [[ "$?" -eq 0 ]]; then
    cat models_new/sentence_segmentation/x* > models_new/sentence_segmentation/model.safetensors
    cat models_new/classification/x* > models_new/classification/model.safetensors
    #rm models/*/x*
fi

pip3 install -r requirements.txt
