#!/bin/bash


echo run task:pipeline_entity_property !!
wandb login d06ca290bf76e5ed0d636a32474b9719e1666f39
python ./pipeline_entity_property.py \
  --base_model kykim/electra-kor-base \
  --do_train \
  --do_eval \
  --run_name 'kyelectra(baseline) train+dev+test552+gold120+crawling1141'\
  --train_data '/home/nlplab'
  --learning_rate 5e-6 \
  --eps 9e-9 \
  --num_train_epochs 30 \
  --entity_property_model_path ../saved_model/kyelectra_train+dev+test552+gold120+crawling1141/category_extraction/ \
  --polarity_model_path ../saved_model/kyelectra_train+dev+test552+gold120+crawling1141/polarity_classification/ \
  --batch_size 64 \
  --max_len 256\
  #--classifier_hidden_size 1024




