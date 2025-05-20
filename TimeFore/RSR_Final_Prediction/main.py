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
import ast
import threading
import math
from concurrent.futures import ThreadPoolExecutor

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SLU')
    parser.add_argument('--task', type=str, default='eval', help='eval or prediction')
    parser.add_argument('--model_name', type=str, default='qwen25-72b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='http://your_api_base_url/v1', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=5, help='Request for the number of concurrent LLMs.')
    return parser

# 使用方法
parser = create_parser()
args = parser.parse_args()
BASE_URL = args.base_url
MODEL_NAME = args.model_name
print(f'The specified model name:{args.model_name}')
OPENAI_API_KEY = args.api_key

def main():
    if args.task == 'prediction':
        CONCURRENT_WORKERS = args.workers
        global_save_lock = threading.Lock()
        llm = ChatOpenAI(
            model_name=MODEL_NAME,          # 对应环境变量 MODEL_NAME
            openai_api_key=OPENAI_API_KEY,  # 对应环境变量 OPENAI_API_KEY
            base_url=BASE_URL               # 对应环境变量 BASE_URL
        )
        
        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        # 定义保存状态的文件名
        state_file = f'./results/{MODEL_NAME} Final Answer Generation Process.txt'
        result_saved_file = f'./results/{MODEL_NAME} Final Answer Generation Results.json'
        
        json_file = f'../TimesNet/results/{MODEL_NAME}\'s Future Price Prediction Results.json'
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
                if data['BERT_pred_query_type'] == "Numerical":
                    prompt = ChatPromptTemplate.from_template(templates['Numerical_Prompt'])
                    chain = prompt | llm
                    predict_final_result(data = data, file_name = result_saved_file, chain = chain, save_lock = global_save_lock)
                elif data['BERT_pred_query_type'] == "Judgment":
                    prompt = ChatPromptTemplate.from_template(templates['Judgment_Prompt'])
                    chain = prompt | llm
                    predict_final_result(data = data, file_name = result_saved_file, chain = chain, save_lock = global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False

        
        with tqdm(total=len(remaining_datas), desc="Processing progress.", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in remaining_datas]
                for future in futures:
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"error: {str(e)}")
                        continue
        print("All data processing is complete!！")

    elif args.task == 'eval':
        result_saved_file = f'./results/{MODEL_NAME} Final Answer Generation Results(After Normalization).json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        trues = []
        predicts = []
        judge_trues = []
        judge_preds = []
        judgmentQuestionCount = 0
        judgmentSqlExecutionFailure  = 0
        pendingJudgmentQuestions = 0
        numericalQuestionCount = 0
        numericalSqlExecutionFailure  = 0
        pendingNumericalQuestions = 0
        for i, data in enumerate(datas):
            if data['answer'] == []:
                data['answer'] = [["无"]]

            if data['query_type']=='Judgment':
                judgmentQuestionCount += 1
                if data['predict_history_SQL_result'] == None:
                    judgmentSqlExecutionFailure += 1
                    pendingJudgmentQuestions += 1
                    continue
                if '<Answer>:' in data['predict_answer']:
                    data['predict_answer'] = extract_answer(data['predict_answer'])
                if ':' in data['predict_answer']:
                    data['predict_answer'] = data['predict_answer'].replace(':', '')
                predict_final_answer = parse_list_string(data['predict_answer'])
                predict_list, true_list,  success = process_judgment_predict_true(predict_final_answer, data['answer'])
                if success == False:
                    pendingJudgmentQuestions += 1
                else:
                    judge_preds.append(set(predict_list))
                    judge_trues.append(set(true_list))


            elif data['query_type']=='Numerical':
                numericalQuestionCount += 1
                if data['predict_history_SQL_result'] == None:
                    numericalSqlExecutionFailure += 1
                    pendingNumericalQuestions += 1
                    continue
                if '<Answer>:' in data['predict_normalization_answer']:
                    data['predict_normalization_answer'] = extract_answer(data['predict_normalization_answer'])
                if ':' in data['predict_normalization_answer']:
                    data['predict_normalization_answer'] = data['predict_normalization_answer'].replace(':', '')
                try:
                    predict_final_answer = parse_list_string(data['predict_normalization_answer']) 
                except:
                    pendingNumericalQuestions += 1
                    continue
                    
                true_list, predict_list, success = process_numerical_predict_true(predict_final_answer, data['answer'])
                if np.any(np.isnan(predict_list)):
                    true_list, predict_list, success = [], [], False
                if success == False:
                    pendingNumericalQuestions += 1
                predicts.extend(predict_list)
                trues.extend(true_list)
        indices_to_remove = [i for i, p in enumerate(predicts) if math.isnan(p)]

        for idx in sorted(indices_to_remove, reverse=True):
            del predicts[idx]
            del trues[idx]

        trues_np = np.array(trues)
        predicts_np = np.array(predicts)
        mae = np.mean(np.abs(predicts_np - trues_np))
        mre = np.mean(np.abs((predicts_np - trues_np) / trues_np))

        mse = np.mean((predicts_np - trues_np) ** 2)
        judgmentQuestionsCompletionRate = (judgmentQuestionCount - pendingJudgmentQuestions) / judgmentQuestionCount
        judgmentSqlExecutionSuccessRate = (judgmentQuestionCount - judgmentSqlExecutionFailure) / judgmentQuestionCount
        numericalQuestionsCompletionRate = (numericalQuestionCount - pendingNumericalQuestions) / numericalQuestionCount
        numericalSqlExecutionSuccessRate = (numericalQuestionCount - numericalSqlExecutionFailure) / numericalQuestionCount
        print(f"Evaluation metrics for Numerical problems(CompletionRate:{numericalQuestionsCompletionRate * 100:.4f}%,SQL执行成功占比:{numericalSqlExecutionSuccessRate * 100:.4f}%):")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"│ MSE │ {mse:.4f} │")
        print(f"│ MAE │ {mae:.4f} │")
        print(f"│ MRE │ {mre:.4f} │")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        print(f"Evaluation metrics for Judgment problems(CompletionRate:{judgmentQuestionsCompletionRate * 100:.4f}%,SQL执行成功占比:{judgmentSqlExecutionSuccessRate * 100:.4f}%):")
        calculate_metrics(judge_trues, judge_preds)

    elif args.task == 'normalization':
        CONCURRENT_WORKERS = args.workers
        global_save_lock = threading.Lock()
        normalization_model = 'c101-qwen25-72b'
        llm = ChatOpenAI(
            model_name=normalization_model,        
            openai_api_key=OPENAI_API_KEY, 
            base_url=BASE_URL           
        
        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        result_saved_file = f'./results/{MODEL_NAME} Final Answer Generation Results(After Normalization).json'
        json_file = f'./results/{MODEL_NAME} Final Answer Generation Results.json'
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
                prompt = ChatPromptTemplate.from_template(templates['Numerical_Normalization_Prompt'])
                chain = prompt | llm
                predict_normalization_result(data = data, file_name = result_saved_file, chain = chain, save_lock = global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False

        
        with tqdm(total=len(remaining_datas), desc="Processing progress", unit="query") as pbar:
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



if __name__ == "__main__":
    main()