import torch
import numpy as np
from transformers import BertTokenizer
import json
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from ipywidgets import FloatProgress
from torch.utils.tensorboard import SummaryWriter
import re
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import json
import pandas as pd

def train(model, train_data, val_data, learning_rate, epochs, writer, tokenizer, batch_size):
    # Sort the two tensors along the specified dimension using the Dataset class to retrieve the training and validation sets
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)
    # Use DataLoader to retrieve data based on batch_size, and shuffle samples during training
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # loss
    criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch_num in range(epochs):
        total_intent_acc_train = 0
        total_loss_train = 0
        total_tokens_train = 0
        for train_input, train_intent_label in tqdm(train_dataloader):
            train_intent_label = train_intent_label.to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # model output
            intent_probability = model(input_id, mask)
            # compute loss
            intent_loss = binary_criterion(intent_probability, train_intent_label.float())
            active_loss = mask.view(-1) == 1
            loss = intent_loss
            total_loss_train += loss.item()
            # compute metric
            intent_acc = compute_multi_label_acc(intent_probability, train_intent_label)
            total_intent_acc_train += intent_acc
            batch_token_nums = active_loss.sum().item()
            total_tokens_train += batch_token_nums
            model.zero_grad()
            loss.backward()
            optimizer.step()
        # ------- val -----------
        total_intent_acc_val = 0
        total_loss_val = 0
        total_tokens_val = 0
        with torch.no_grad():
            for val_input, val_intent_label in val_dataloader:
                val_intent_label = val_intent_label.to(device)
                mask = val_input['attention_mask'].squeeze(1).to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                intent_probability = model(input_id, mask)
                intent_loss = binary_criterion(intent_probability, val_intent_label.float())
                active_loss = mask.view(-1) == 1
                # compute intent num loss
                loss = intent_loss
                total_loss_val += loss.item()
                # compute metric
                intent_acc = compute_multi_label_acc(intent_probability, val_intent_label)
                total_intent_acc_val += intent_acc
                batch_token_nums = active_loss.sum().item()
                total_tokens_val += batch_token_nums
        writer.add_scalar('Loss/train', total_loss_train / len(train_data), epoch_num)
        writer.add_scalar('Accuracy/train_intent', total_intent_acc_train / len(train_data), epoch_num)
        writer.add_scalar('Loss/val', total_loss_val / len(val_data), epoch_num)
        writer.add_scalar('Accuracy/val_intent', total_intent_acc_val / len(val_data), epoch_num)

        print(
            f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Intent Accuracy: {total_intent_acc_train / len(train_data): .3f}
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Intent Accuracy: {total_intent_acc_val / len(val_data): .3f} ''')
        writer.close()


def evaluate(model, datas, all_intents, tokenizer, device):
    # Initialize storage variables
    pred_intent_label = []
    true_intent_label = []

    for data in tqdm(datas):
        query = data['query']
        intent = data['query_type']
        intent_label = intent2label(intent).index(1)
        true_intent_label.append(intent_label)


        encoded_text = tokenizer(query, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
        new_tokens = align_tokens_with_query(tokens, query)

        encoded_text_id = encoded_text['input_ids'].to(device)
        mask = encoded_text['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(encoded_text_id, mask)

        intent_probility = outputs[0].view(-1)
        _, intent_idx = torch.topk(intent_probility, k=1, dim=0)
        intent_idx = intent_idx.cpu()
        intent = all_intents[intent_idx[0]]
        data["BERT_pred_query_type"] = intent
        pred_intent_label.append(intent_idx)
        
    print('### Query Type Classification Report ###:')
    inetnt_report = sk_classification_report(true_intent_label, pred_intent_label, digits=4)
    print(inetnt_report)

    return datas

def find_key(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

def restore_keywords_from_tokens(tokens, token_slot):
    keywords = []
    current_tokens = []
    current_label = None
    token_slot = token_slot[1:-1]

    for token, slot in zip(tokens, token_slot):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))
    keyword_pair = []
    for keyword in keywords:
        if keyword[-1] == 'city':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'district':
            keyword_pair.append(f'区域:{keyword[0]}')
        elif keyword[-1] == 'community':
            keyword_pair.append(f'项目名:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'年份:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'月份:{keyword[0]}')
        elif keyword[-1] == 'time':
            keyword_pair.append(f'时间:{keyword[0]}')

    return keyword_pair


def align_tokens_with_query(tokens, query):
    query = list(query)
    new_tokens = []
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        elif token == query[0]:
            new_tokens.append(token)
            query = query[1:]
        elif '##' in token:
            token = token[2:]
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    
        elif '[UNK]' == token:
            end_index = query.index(tokens[i+1])
            unk = ''.join(query[0:end_index])
            new_tokens.append(unk)
            query = query[end_index:]

        elif len(token)>1:
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
    return new_tokens


def num_2_slots(num_list, slots_num):
    # 创建一个反向映射字典
    num_to_slot = {v: k for k, v in slots_num.items()}
    # 将数字转换为对应的标签
    return [num_to_slot[num] for num in num_list]


def intent2label(intents_row):
    #intents_num = {'均价趋势预测': 0, '均价点时间预测': 1, '均价片时间预测': 2}
    intents_num = {"Numerical": 0, "Judgment": 1}
    intents_label = [0] * len(intents_num)
    if '+' in intents_row:
        intents = intents_row.split('+')
        for intent in intents:
            intents_label[intents_num[intent]] = 1
    elif '+' not in intents_row:
        intent = intents_row
        intents_label[intents_num[intent]] = 1
    return intents_label

def pad_to_512(input_string, max_pad_lenth=512):
    while len(input_string) < max_pad_lenth:
        input_string.append(int(-100))
    return input_string


def json2dataframe(all_datas, tokenizer, intents_num):
    df = pd.DataFrame(columns=['category', 'text'])
    for data in all_datas:
        query = data['query']
        encoded_text = tokenizer(query, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
        intents_label = [0.0] * len(intents_num)
        intent = data['query_type']
        intents_label[intents_num[intent]] = 1.0
        df = pd.concat([df, pd.DataFrame([{'category': intents_label, 'text': data['query']}])], ignore_index=True)
    return df

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.labels = df['category']
        self.texts = [tokenizer(text, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]


    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def tensors_equal_ignore_order(tensor1, tensor2):
    # Sort the two tensors along the specified dimension
    sorted_tensor1, _ = torch.sort(tensor1)
    sorted_tensor2, _ = torch.sort(tensor2)
    results = []
    for row1, row2 in zip(sorted_tensor1, sorted_tensor2):
        results.append(torch.equal(row1, row2))
        results_tensor = torch.tensor(results, dtype=torch.bool)
    return results_tensor

# Input two tensors: the first tensor is the probability tensor, and the second tensor is the one-hot encoded label tensor.
def compute_multi_label_acc(probility, label):
    probility, idx1 = torch.sort(probility, descending=True)
    label, idx2 = torch.sort(label, descending=True)
    idx1 = idx1[:,0:2]
    idx2 = idx2[:,0:2]
    for i,labl in enumerate(label):
        if labl.sum() < 2:
            idx1[i,1] = 0
            idx2[i,1] = 0
    acc = tensors_equal_ignore_order(idx1, idx2).sum().item()
    return acc