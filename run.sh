#!/bin/bash

python train.py --model_name=fm --max_epochs=50 --intermediate_dim=16 --dropout=0.1 --multi_class=True
python eval.py --model_names=fm --multi_class=True

python train.py --model_name=fb --max_epochs=20 --intermediate_dim=16 --dropout=0.1
python eval.py --model_names=fb