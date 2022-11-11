import os
import random
import logging

from datetime import datetime

import ntpath
import argparse

import json

import torch
import numpy as np

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    RobertaModel,
    ElectraModel,
    AutoTokenizer,
    AutoModel,
    ElectraForQuestionAnswering
)
'''
##util-functions for belows##
- model config infos: MODEL_CLASSES, MODEL_PATH_MAP, SPECIAL_TOKENS, TOKEN_MAX_LENGTH
- metric functions: MCC, compute_metrics, ...
- path functions: getParentPath
- save/load model_path functions: save_model, load_model
- token functions: getTokLength
'''


MODEL_CLASSES = {
    'roberta-base': (AutoConfig, RobertaModel, AutoTokenizer),
    'koelectra': (ElectraConfig, ElectraModel, ElectraTokenizer),
    'koelectraQA': (ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer),
    'koelectra_tunib':(AutoConfig, AutoModel, AutoTokenizer),
    'klue/roberta-large':(AutoConfig, AutoModel, AutoTokenizer),
    'kcelectra':(AutoConfig, AutoModel, AutoTokenizer),
    'mdeberta':(AutoConfig, AutoModel, AutoTokenizer),
    'kyelectra':(ElectraConfig, ElectraModel, ElectraTokenizer),
    'koelectra-v3':(ElectraConfig, ElectraModel, ElectraTokenizer),
    'test':(AutoConfig, AutoModel, AutoTokenizer)
    
}

MODEL_PATH_MAP = {
    'koelectra': 'monologg/koelectra-base-v3-discriminator',
    'roberta-base': 'klue/roberta-base',
    'koelectra_tunib': 'tunib/electra-ko-base',
    'klue/roberta-large': 'klue/roberta-large',
    'kcelectra': 'beomi/KcELECTRA-base-v2022',
    'mdeberta': 'lighthouse/mdeberta-v3-base-kor-further',
    'kyelectra': 'kykim/electra-kor-base',
    'test': 'BM-K/KoSimCSE-roberta'
}

DATASET_PATHS = {
    'ABSA' : ('task_ABSA/','nikluge-sa-2022-train.jsonl', 'nikluge-sa-2022-dev.jsonl', 'nikluge-sa-2022-test.jsonl')
} #train, dev ,test, test_labeled
##COLA test_label 바꿈. 

SPECIAL_TOKENS_NUM = {
    'koelectra': (2, 3, 1),
    'roberta-base': (0, 2, 1),
    'koelectra_tunib': (2, 3, 1),
    'kobert': (2, 3, 1),
    'koelectraQA': (2, 3, 1),
} #token_num(CLS,SEP,PAD): koelectra(CLS=2,SEP=3,PAD=1), roberta(CLS=0,SEP=2,PAD=1)
#kobert 확인하기.

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
} #WiC, 

TOKEN_MAX_LENGTH = {
    'ABSA' : 256,

} 


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seedNum, device):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)


def parse_args():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--train_data", type=str, default="../data/input_data_v1/train.json",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/input_data_v1/test.json",
        help="test file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/input_data_v1/dev.json",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="kclectra"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )

    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    parser.add_argument(
        "--pred_data_path", type=str
    )
    parser.add_argument(
        "--load_pred_data", type=str
    )
    
    args = parser.parse_args()
    return args


##get Path functions
def path_fpath(path):
    fpath, fname = ntpath.split(path)
    return fpath #fpath or ntpath.basename(fname)
def path_leaf(path):
    fpath, fname = ntpath.split(path)
    return ntpath.basename(fname) #fpath or ntpath.basename(fname)
def getFName(fname):
    fname_split = fname.split('.') #name, extenstion
    new_fname=fname_split[0]#+'.jpg'
    return new_fname

##get parent/home directory path##
def getParentPath(pathStr):
    return os.path.abspath(pathStr+"../../")
#return parentPth/parentPth of pathStr -> hdd1/
def getHomePath(pathStr):
    return getParentPath(getParentPath(getParentPath(pathStr))) #ast/src/

def print_timeNow():
    cur_day_time = datetime.now().strftime("%m/%d, %H:%M:%S") #Date %m/%d %H:%M:%S
    return cur_day_time

def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
        
# json list를 jsonl 형태로 저장
def jsonldump(j_list, fname):
    f = open(fname, "w", encoding='utf-8')
    for json_data in j_list:
        f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list




