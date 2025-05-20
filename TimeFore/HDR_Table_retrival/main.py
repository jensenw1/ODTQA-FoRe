import numpy as np
import pandas as pd
from ipywidgets import FloatProgress
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import json
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from tqdm import tqdm
import ast
import re
from rank_bm25 import BM25Okapi
import jieba
from utils import *
import argparse
from rank_bm25 import BM25Okapi
import jieba
import threading
from concurrent.futures import ThreadPoolExecutor

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SLU')
    parser.add_argument('--task', type=str, default='prediction', help='eval or prediction')
    parser.add_argument('--model_name', type=str, default='your_model_name', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='http://your_model_path/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='LLM API KEY')
    parser.add_argument('--processCount', type=int, default=1, help='The number of processes sending requests to the LLM.')
    parser.add_argument('--workers', type=int, default=5, help='Request for the number of concurrent LLMs.')
    return parser

# 使用方法
parser = create_parser()
args = parser.parse_args()
BASE_URL = args.base_url
MODEL_NAME = args.model_name
OPENAI_API_KEY = args.api_key

def main():
    if args.task == 'prediction':
        CONCURRENT_WORKERS = args.workers 
        global_save_lock = threading.Lock()
        print(f'Connecting to LLM-{args.model_name}...')

        llm = ChatOpenAI(
            model_name=MODEL_NAME,         
            openai_api_key=OPENAI_API_KEY,  
            base_url=BASE_URL              
        )
        
        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)

        result_saved_file = f'./results/{MODEL_NAME} Table Caption Prediction Results.json' # This .json file is used to record the output results of the LLM.
        json_file = '../RSR_BERT/results/test-with-BERT_pred_query_type.json'
        with open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        processed_queries = set()
        if os.path.exists(result_saved_file):
            try:
                with open(result_saved_file, 'r') as f:
                    existing_data = json.load(f)
                    processed_queries = {item['query'] for item in existing_data if 'query' in item}
            except json.JSONDecodeError:
                pass
        # Filter unprocessed data.
        remaining_datas = [data for data in datas if data.get('query') not in processed_queries]
        # Multithreading processing function.
        def process_data(data):
            try:
                #intent = data['BERT_pred_intent']
                prompt = ChatPromptTemplate.from_template(templates['Summary_Prompt'])
                chain = prompt | llm
                predict_table_caption(data = data, file_name = result_saved_file, chain = chain, save_lock = global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False
        # Use a thread pool for parallel processing.
        
        with tqdm(total=len(remaining_datas), desc="Processing progress", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in remaining_datas]
                for future in futures:
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        continue
        print("Complated！")



    elif args.task == 'eval':
        table_names_file = '../../datasets/all_table_caption'
        with open(table_names_file, 'r', encoding='utf-8') as f:
            all_table_captions = json.load(f)
        
        corpus = all_table_captions
        tokenized_corpus = [list(jieba.cut(doc, cut_all=True)) for doc in corpus]
        
        bm25 = BM25Okapi(tokenized_corpus)

        result_saved_file = f'./results/{MODEL_NAME} Table Caption Prediction Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        predicts = []
        bm_25s = []
        trues = []

        for data in datas:
            predict = data['predict_table_caption']
            if '<Summary>:' in data['predict_table_caption']:
                predict = data['predict_table_caption'].split("<Summary>:")[1].strip()

            if ',' in data['predict_table_caption'] and '[' in data['predict_table_caption']:
                predict = predict.split(',')
                #print(predict)
                bm_25 = []
                for i, pred in enumerate(predict):
                    pred = pred.replace("[", "")
                    pred = pred.replace("]", "")
                    pred = pred.replace("'", "")
                    pred = pred.replace('"', '')
                    pred = pred.replace('\\', "")
                    pred = pred.replace("\"", "")
                    pred = pred.replace("\'", "")
                    pred = pred.replace("\n", "")
                    pred = pred.replace(" ", "")
                    pred = pred.replace("<Summary>:", "")
                    predict[i] = pred
                    if pred not in all_table_captions:
                        tokenized_query = list(jieba.cut(pred, cut_all=True))
                        result = bm25.get_top_n(tokenized_query, corpus, n=1)
                        bm_25.append(result[0])
                    else:
                        bm_25.append(pred)
                predict = list(set(predict))
                bm_25 = list(set(bm_25))
        
            predicts.append(predict)
            bm_25s.append(bm_25)
            true = data['table_caption']
            trues.append(true)
            data['predict_table_caption(BM25)'] = bm_25
        print('1、The accuracy of LLM output.')
        metric_compute(trues, predicts)
        print('2、The accuracy of after BM25.')
        metric_compute(trues, bm_25s)
        with open(result_saved_file, 'w', encoding='utf-8') as f:
            json.dump(datas, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()