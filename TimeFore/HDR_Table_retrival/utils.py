import json
import os
import threading

# # Store the generated table names into a JSON file.
class PredictResultStorage:
    def __init__(self, file_name='testdata.json', save_lock=None):
        self.current_data = {}
        self.file_name = file_name
        self.save_lock = save_lock or threading.Lock()
    def set_data(self, data_dict):
        self.current_data.update(data_dict)
    def save_data(self):
        with self.save_lock:
            if os.path.exists(self.file_name):
                with open(self.file_name, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(self.current_data)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            else:
                with open(self.file_name, 'w') as f:
                    json.dump([self.current_data], f, indent=4)
            self.current_data = {}

def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]


def extract_content_after_summary(response):
    keyword = '<Summary>:'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]

def predict_table_caption(data: dict, file_name: str, chain, save_lock):
    query = data['query']
    try:
        response = chain.invoke({"query": query})
    except Exception as e:  
        print(f"API call failed, save operation has been skipped | Error details: {str(e)}")
        return
    predicted_table_name = response.content
    if '</think>' in predicted_table_name:
        predicted_table_name = extract_content_after_think(predicted_table_name)
    if '<Summary>:' in predicted_table_name:
        predicted_table_name = extract_content_after_summary(predicted_table_name)
    data["predict_table_caption"] = predicted_table_name
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()




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

    return keywords

def restore_keywords_from_query(query, slots):
    keywords = []
    current_tokens = []
    current_label = None
    query = list(query)
    if slots[0] == '[CLS]':
        slots = slots[1:-1]

    for token, slot in zip(query, slots):
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

    return keywords


def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengthes not equal!'
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    # Convert all list elements to a list.
    for true_label, pred_label in zip(trues, preds):
        # Convert all list elements to a list.
        if isinstance(true_label, type('')):
            true_label = [true_label]
        if isinstance(pred_label, type('')):
            pred_label = [pred_label]
        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    P = precision/precision_all
    R = recall/recall_all
    F1 = 2 * P * R / (P + R)
    print("Evaluation Metrics:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"│ Precision (P) │ {P:.4f} │")
    print(f"│ Recall (R)    │ {R:.4f} │")
    print(f"│ F1 Score      │ {F1:.4f} │")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return P, R, F1
