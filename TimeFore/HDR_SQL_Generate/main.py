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
from tabulate import tabulate
import jieba
import psycopg2
from psycopg2 import sql, DatabaseError
from utils import *
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SLU')
    parser.add_argument('--task', type=str, default='test', help='transform, eval or prediction')
    parser.add_argument('--model_name', type=str, default='qwen25-72b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='http://your_base_url/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=5, help='Request for the number of concurrent LLMs.')
    return parser

parser = create_parser()
args = parser.parse_args()

BASE_URL = args.base_url
MODEL_NAME = args.model_name
OPENAI_API_KEY = args.api_key
CONCURRENT_WORKERS = args.workers

global_save_lock = threading.Lock()
def main():
    if args.task == 'prediction':
        print(f'Connecting to LLM-{args.model_name}...')
        llm = ChatOpenAI(
            model_name=MODEL_NAME,         
            openai_api_key=OPENAI_API_KEY,  
            base_url=BASE_URL           
        )

        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)


        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        json_file = f'../HDR_Table_retrival/results/{MODEL_NAME} Table Caption Prediction Results.json'
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

        remaining_datas = [data for data in datas if data.get('query') not in processed_queries]


        def process_data(data):
            try:
                prompt = ChatPromptTemplate.from_template(templates['SQL_Prompt'])
                chain = prompt | llm
                predict_SQL(data=data, file_name=result_saved_file, chain=chain, save_lock = global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False

        with tqdm(total=len(remaining_datas), desc="处理进度", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in remaining_datas]
                for future in futures:
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"error: {str(e)}")
                        continue
        print("All data processing is complete!")

    elif args.task == 'transform':
        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        # 生成24个月的时间轴（2022年1月 到 2023年12月）
        months = [f"{year}年{month}月" for year in [2022, 2023] for month in range(1, 13)]
        # 处理每个数据条目
        for data in tqdm(datas):
            result = {}
            if data['predict_history_SQL_result'] == None:
                data['llm_searched_history_price_time_series'] = None
                continue
            else:
                for entry in data['predict_history_SQL_result']:
                    project = None
                    time_str = None
                    price = None
                    for element in entry:
                        if isinstance(element, (int, float)):
                            price = element
                        elif isinstance(element, str):
                            if '年' in element and '月' in element:
                                time_str = element
                            else:
                                project = element
                    
                    if project is None:
                        project = 'single'
                    
                    if time_str and price is not None:
                        if time_str in months:
                            idx = months.index(time_str)
                            if project not in result:
                                result[project] = [0] * 24
                            result[project][idx] = price
                        else:
                            result = None
                            continue

                data['llm_searched_history_price_time_series'] = result
        with open(result_saved_file, 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)

        
    elif args.task == 'eval':
        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        preds = []
        trues = []
        unexecutable_sql = 0
        all_sql = 0
        for result in datas:
            if result['predict_history_SQL_result'] == None:
                result['predict_history_SQL_result'] = []
                unexecutable_sql += 1
            preds.append(result['predict_history_SQL'])
            trues.append(result['history_SQL'])
            all_sql += 1
        ECR = (all_sql - unexecutable_sql)/all_sql
        table = [
            ["ECR", f"{ECR:.4f}"]
        ]
        
        #print(f"{result_saved_file}")
        print(tabulate(table, tablefmt="grid"))

        
        uuids = {}
        for result in datas:
            table_results = result['predict_history_SQL_result']
            true_SQL_answer = result['history_SQL_result']
            if table_results is not None and all(isinstance(i, list) for i in table_results):
                table_results = [tuple(sublist) for sublist in table_results]
            if true_SQL_answer is not None and all(isinstance(i, list) for i in true_SQL_answer):
                true_SQL_answer = [tuple(sublist) for sublist in true_SQL_answer]
            if table_results != None and set(table_results) == set(true_SQL_answer) and len(table_results) == len(true_SQL_answer):
                result['predict_correctness'] = True
            else:
                result['predict_correctness'] = False
            if result['uuid'] not in uuids:
                uuids[result['uuid']] = []
            uuids[result['uuid']].append(result['predict_correctness'])
        
        
        for key in uuids.keys():
            c = sum(uuids[key])
            n = len(uuids[key])
            uuids[key] = {'c': c, 'n': n}
            uuids[key]['pass1'] = pass_at_k(n = uuids[key]['n'], c = uuids[key]['c'],k=1)

        print(f'Overallpass@1：{calculate_average_pass1(uuids)}')



if __name__ == "__main__":
    main()