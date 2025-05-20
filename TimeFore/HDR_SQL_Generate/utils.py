import json
import os
import psycopg2
from psycopg2 import sql, DatabaseError
import numpy as np
import threading

class PostgresQueryExecutor:
    def __init__(self, host='127.0.0.1', database='历史数据', user='postgre', password='your_database_password', port="5432"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cur = self.conn.cursor()
        except DatabaseError as e:
            print(f"Database connection error: {e}")
            raise

    def execute_sql(self, sql_statement):
        if not self.conn or not self.cur:
            self.connect()
        try:
            self.cur.execute(sql_statement)
            result = self.cur.fetchall() 
            headers = [desc[0] for desc in self.cur.description]
            self.conn.commit()
            
            return headers, result
        except Exception as e:
            print(f"Error executing SQL statement: {e}")
            return None, None

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()


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


def predict_SQL(data: dict, file_name: str, chain, save_lock):
    query = data['query']
    table_caption = data['predict_table_caption(BM25)']
    try:
        response = chain.invoke({"query":query, "table_caption":table_caption})
        sql_statement = response.content
        if '</think>' in sql_statement:
            sql_statement = extract_content_after_think(sql_statement)
        if ' <SQL>:' in sql_statement or '<SQL>:' in sql_statement:
            sql_statement = extract_sql_content(sql_statement)
        data["predict_history_SQL"] = sql_statement

        executor = PostgresQueryExecutor(database='历史数据')
        table_heads, table_results = executor.execute_sql(sql_statement)
        data['predict_history_SQL_result'] = table_results
        executor.close()
    except Exception as e:
        print(f"API call failed, save operation has been skipped | Error details: {str(e)}")
        return
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()

def extract_sql_content(input_string):
    # Try with both possible prefixes
    prefixes = ['<SQL>:', ' <SQL>:']
    
    for prefix in prefixes:
        start_index = input_string.find(prefix)
        if start_index != -1:
            # Add the length of the prefix to get to the start of the SQL content
            start_index += len(prefix)
            sql_content = input_string[start_index:]
            return sql_content


def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengthes not equal!'
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    accuracy = 0
    acc_all = 0
    for true_label, pred_label in zip(trues, preds):
        if isinstance(true_label, type('')):
            true_label = [true_label]
        if isinstance(pred_label, type('')):
            pred_label = [pred_label]
        if true_label == pred_label:
            accuracy += 1
        acc_all += 1
        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    acc = accuracy/acc_all
    P = precision/precision_all
    R = recall/recall_all
    F1 = 2 * P * R / (P + R)
    return acc, P, R, F1



def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]



def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k /np.arange(n - c + 1, n + 1))

def calculate_average_pass1(data):
    total_pass1 = sum(item['pass1'] for item in data.values())
    count = len(data)
    if count == 0:
        return 0 
    return total_pass1 / count