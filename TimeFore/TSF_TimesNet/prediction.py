import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import random
import argparse
import os
import torch
import random
import numpy as np
from model import *
from Embed import *
from utils import *
import json

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--task_name', type=str, default='short_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name, options: [Autoformer, Transformer, TimesNet]')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    
    # model define
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--seed', type=int, default=2025, help='number of seed')
    parser.add_argument('--epoch', type=int, default=300, help='num of epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--path_of_imputation_datasets', type=str, default='./pridict_price_datasets', help='imputation datasets path')
    parser.add_argument('--input_file', type=str, help='输入文件路径（支持.txt或.json）')
    parser.add_argument('--model_name', type=str, default='Qwen', help='model name')

    return parser


criterion = nn.MSELoss(reduction='sum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if prediction_args.task == 'train':
        train_loader = load_prediction_price_dataset(prediction_args.path_of_imputation_datasets, 'train', batch_size=prediction_args.batch_size)
        val_loader = load_prediction_price_dataset(prediction_args.path_of_imputation_datasets, 'val', batch_size=prediction_args.batch_size)

        fix_seed = prediction_args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        model = TimesNet(prediction_args).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=prediction_args.learning_rate)


        num_epochs = prediction_args.epoch
        train_losses, val_losses, val_mae, val_mse, best_model_state = train_prediction_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
        # Save the best model parameters
        model_save_path = './pretrain_args/best_TimesNet_Predition_model_without_district.pth'
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

        save_args(prediction_args, './pretrain_args/best_TimesNet_Predition_model_without_district_args.json')

    if prediction_args.task == 'test':
        test_loader = load_prediction_price_dataset(prediction_args.path_of_imputation_datasets, 'test', batch_size=prediction_args.batch_size)
        test_loss = 0  
        all_inputs = [] 
        all_pred = []  
        all_target = []
        all_nan_mask = []
        all_target_mask = []
        prediction_model = TimesNet(prediction_args).to(device)
        prediction_model_path = './pretrain_args/best_TimesNet_Predition_model_without_district.pth'
        prediction_model.load_state_dict(torch.load(prediction_model_path))
        prediction_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
                outputs = prediction_model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None).squeeze(-1)
                masked_outputs = outputs * target_mask
                masked_targets = targets * target_mask
                loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
                test_loss += loss.item()
                all_inputs.append(x_enc.cpu().numpy())  # 保存输入部分
                all_pred.append(outputs.cpu().numpy())  
                all_target.append(targets.cpu().numpy())
                all_nan_mask.append(nan_mask.cpu().numpy())
                all_target_mask.append(target_mask.cpu().numpy())

        all_inputs = np.vstack(all_inputs)      
        all_pred = np.vstack(all_pred)     
        all_target = np.vstack(all_target) 
        all_nan_mask = np.vstack(all_nan_mask)    
        all_target_mask = np.vstack(all_target_mask)

        full_pred = all_pred[all_target_mask]
        full_target= all_target[all_target_mask]


        mae = np.mean(np.abs(full_pred - full_target))
        mse = np.mean((full_pred - full_target) ** 2)

        print("-----------Metrics after inverse transformation-----------")
        print(f'Test Restored MSE: {mse}')
        print(f'Test Restored MAE: {mae}')



    elif prediction_args.task == 'application':
        with open(prediction_args.input_file) as f:
            inputs = json.load(f)
        print('Imputation and prediction are in progress... Please wait.')
        for input in tqdm(inputs):
            predict_price = {}
            ts = input['llm_searched_history_price_time_series']
            if ts == None:
                input[f'{prediction_args.model}_predict_price'] = None
                continue
            else:
                for project, ts_input in ts.items():
                    ts_input = [np.nan if x == 0 else x for x in ts_input]


                    ts_input = torch.tensor(ts_input, dtype=torch.float32).unsqueeze(0)
                    target_mask = torch.isnan(ts_input)
                    completed_inputs = load_and_run_model(ts_input, nan_mask=torch.zeros_like(ts_input, dtype=torch.bool), target_mask=target_mask, device=device)

                    prediction_model = TimesNet(prediction_args).to(device)
                    model_path = './pretrain_args/best_TimesNet_Predition_model_without_district.pth'
                    prediction_model.load_state_dict(torch.load(model_path))
                    prediction_model.eval()
                    with torch.no_grad():
                        final_outputs = prediction_model(completed_inputs, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None).squeeze(-1)
                    predict_price[project] = final_outputs.detach().cpu().numpy().tolist()
                input[f'{prediction_args.model}_predict_price'] = convert_property_data(predict_price)
                
        result_save_file = f'./results/{prediction_args.model_name}\'s Future Price Prediction Results.json'
        with open(result_save_file, 'w', encoding='utf-8') as f:
            json.dump(inputs, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    parser = create_parser()
    prediction_args = parser.parse_args()
    main()