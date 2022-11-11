import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from pprint import pprint
from datetime import datetime

from transformers import AdamW
from tqdm import trange
from utils import compute_metrics,  MCC, get_label, set_seed, jsonlload, parse_args
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from kobert_datasets import entity_property_pair
from kobert_datasets import special_tokens_dict
import wandb
import copy


from kobert_datasets import get_dataset
from kobert_models import model_ABSA

from transformers import AutoTokenizer

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
data_path=os.getcwd()+'/../../dataset/'
model_name = 'kykim/electra-kor-base' #'kobert', 'roberta-base', 'koelectra'
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.
model_path = MODEL_PATH_MAP[model_name]
label_id_to_name = ['True', 'False']
polarity_id_to_name = ['positive', 'negative', 'neutral']

args=parse_args()

wandb.init(project="Yaechan", entity="nlpteam2", name=args.run_name)


def evaluation(y_true, y_pred, label_len,eval):
    count_list = [0]*label_len
    hit_list = [0]*label_len
    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1
    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i]/count_list[i])

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    if (eval==0):
        print(count_list)
        print(hit_list)
        print(hit_list[0],",",count_list[0],",",hit_list[0]+count_list[1]-hit_list[1])
        print(acc_list)
        print('Entity_Property_accuracy: ', (sum(hit_list) / sum(count_list)))
        wandb.log({"Entity_Property_accuracy": (sum(hit_list) / sum(count_list))})
        print('Entity_Property_macro_accuracy: ', sum(acc_list) / 2)
        wandb.log({"Entity_Property_macro_accuracy": sum(acc_list) / 2})
        # print(y_true)

        y_true = list(map(int, y_true))
        y_pred = list(map(int, y_pred))

        print('Entity_Property_f1_score: ', f1_score(y_true, y_pred, average=None))
        print('Entity_Property_f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
        print('Entity_Property_f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))
    
        wandb.log({"Entity_Property_f1_score": f1_score(y_true, y_pred, average=None)})
        wandb.log({"Entity_Property_f1_score_micro": f1_score(y_true, y_pred, average='micro')})
        wandb.log({"Entity_Property_f1_score_macro": f1_score(y_true, y_pred, average='macro')})
    else:
        print(count_list)
        print(hit_list)
        print(acc_list)
        print('Polarity_accuracy: ', (sum(hit_list) / sum(count_list)))
        wandb.log({"Polarity_accuracy": (sum(hit_list) / sum(count_list))})
        print('Polarity_macro_accuracy: ', sum(acc_list) / 3)
        wandb.log({"Polarity_macro_accuracy": sum(acc_list) / 3})
        print('Polarity_f1_score: ', f1_score(y_true, y_pred, average=None))
        print('Polarity_f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
        print('Polarity_f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))
    
        wandb.log({"Polarity_f1_score": f1_score(y_true, y_pred, average=None)})
        wandb.log({"Polarity_f1_score_micro": f1_score(y_true, y_pred, average='micro')})
        wandb.log({"Polarity_f1_score_macro": f1_score(y_true, y_pred, average='macro')})
        
def train_sentiment_analysis(args):
    if not os.path.exists(args.entity_property_model_path):
        os.makedirs(args.entity_property_model_path)
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)
    homePth = getParentPath(os.getcwd())
    print('homePth:',homePth,', curPth:',os.getcwd())

    tsvPth_train = args.train_data  
    tsvPth_dev ='/home/nlplab/hdd1/Team2/dataset/task_ABSA/nikluge-sa-2022-dev.jsonl'
    #tsvPth_dev = '/home/nlplab/hdd1/Team2/src2/wandb/latest-run/dev(0.3).jsonl'

    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed  고정.

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    entity_property_train_data, polarity_train_data = get_dataset(tsvPth_train, tokenizer, args.max_len)
    entity_property_dev_data, polarity_dev_data = get_dataset(tsvPth_dev, tokenizer, args.max_len)
    
    entity_property_train_dataloader = DataLoader(entity_property_train_data, shuffle=True,
                                  batch_size=args.batch_size)
    entity_property_dev_dataloader = DataLoader(entity_property_dev_data, shuffle=True,
                                batch_size=args.batch_size)
    polarity_train_dataloader = DataLoader(polarity_train_data, shuffle=True,
                                                  batch_size=args.batch_size)
    polarity_dev_dataloader = DataLoader(polarity_dev_data, shuffle=True,
                                                batch_size=args.batch_size)
    
    print('loading model')
    entity_property_model = model_ABSA(args, len(label_id_to_name), len(tokenizer))
    entity_property_model.to(device)
    
    print('end loading')
    
    FULL_FINETUNING = True
    
    if FULL_FINETUNING:
        entity_property_param_optimizer = list(entity_property_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        entity_property_optimizer_grouped_parameters = [
            {'params': [p for n, p in entity_property_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in entity_property_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        entity_property_param_optimizer = list(entity_property_model.classifier.named_parameters())
        entity_property_optimizer_grouped_parameters = [{"params": [p for n, p in entity_property_param_optimizer]}]

    entity_property_optimizer = AdamW(
        entity_property_optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(entity_property_train_dataloader)

    entity_property_scheduler = get_linear_schedule_with_warmup(
        entity_property_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print('[Training Phase]')
    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        epoch_step += 1
    
        entity_property_model.train()
        

        # entity_property train
        entity_property_total_loss = 0

        for step, batch in enumerate(entity_property_train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            entity_property_model.zero_grad()

            loss, _ = entity_property_model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            entity_property_total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=entity_property_model.parameters(), max_norm=max_grad_norm)
            entity_property_optimizer.step()
            entity_property_scheduler.step()

        avg_train_loss = entity_property_total_loss / len(entity_property_train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = args.entity_property_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(entity_property_model.state_dict(), model_saved_path)

        if args.do_eval:
            entity_property_model.eval()

            pred_list = []
            label_list = []

            for batch in entity_property_dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = entity_property_model(b_input_ids, b_input_mask, b_labels)

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list, len(label_id_to_name),0)


    print("training is done")

def predict_from_korean_form(tokenizer, ce_model, pc_model, data):
    
    ce_model.to(device)
    ce_model.eval()
    count = 0
    for sentence in data:
        form = sentence['sentence_form']
        sentence['annotation'] = []
        count += 1
        if type(form) != str:
            print("form type is arong: ", form)
            continue
        for pair in entity_property_pair:
            tokenized_data = tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)

            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
            with torch.no_grad():
                _, ce_logits = ce_model(input_ids, attention_mask)

            ce_predictions = torch.argmax(ce_logits, dim = -1)

            ce_result = label_id_to_name[ce_predictions[0]]

            if ce_result == 'True':
                with torch.no_grad():
                    _, pc_logits = pc_model(input_ids, attention_mask)

                pc_predictions = torch.argmax(pc_logits, dim=-1)
                pc_result = polarity_id_to_name[pc_predictions[0]]

                sentence['annotation'].append([pair, pc_result])


    return data
    
def evaluation_f1(true_data, pred_data):
    
    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):

        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano  in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano  in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

    ce_precision = ce_eval['TP']/(ce_eval['TP']+ce_eval['FP'])
    ce_recall = ce_eval['TP']/(ce_eval['TP']+ce_eval['FN'])

    ce_result = {
        'Precision': ce_precision,
        'Recall': ce_recall,
        'F1': 2*ce_recall*ce_precision/(ce_recall+ce_precision)
    }

    pipeline_precision = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FP'])
    pipeline_recall = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FN'])

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': 2*pipeline_recall*pipeline_precision/(pipeline_recall+pipeline_precision)
    }

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }


def test_sentiment_analysis(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = jsonlload(args.test_data)

    entity_property_test_data, polarity_test_data = get_dataset(test_data, tokenizer, args.max_len)
    entity_property_test_dataloader = DataLoader(entity_property_test_data, shuffle=True,
                                batch_size=args.batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data, shuffle=True,
                                                  batch_size=args.batch_size)
    
    model = model_ABSA(args, len(label_id_to_name), len(tokenizer))
    model.load_state_dict(torch.load(args.entity_property_model_path, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = model_ABSA(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.load_state_dict(torch.load(args.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(tokenizer, model, polarity_model, copy.deepcopy(test_data))

    # jsondump(pred_data, './pred_data.json')
    # pred_data = jsonload('./pred_data.json')

    print('F1 result: ', evaluation_f1(test_data, pred_data))

    pred_list = []
    label_list = []
    print('polarity classification result')
    for batch in polarity_test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

        predictions = torch.argmax(logits, dim=-1)
        pred_list.extend(predictions)
        label_list.extend(b_labels)

    evaluation(label_list, pred_list, len(polarity_id_to_name))

    
if __name__ == "__main__":

    if args.do_train:
        train_sentiment_analysis(args)
    # elif args.do_demo:bash t
    #     demo_sentiment_analysis(args)
    elif args.do_test:
        test_sentiment_analysis(args)
    


