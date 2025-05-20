import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import random
import argparse
import json
import os
import torch
import random
import numpy as np
from model import *
from Embed import *
from utils import *

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--task_name', type=str, default='imputation', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    
    # model define
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--epoch', type=int, default=300, help='num of epoch')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--seed', type=int, default=2025, help='number of seed')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--path_of_imputation_datasets', type=str, default='./price_imputation_datasets', help='imputation datasets path')
    return parser

# 使用方法
parser = create_parser()
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss(reduction='sum')
def main():
    if args.task == 'train':
        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        # To load training data
        train_loader = load_imputation_price_dataset(args.path_of_imputation_datasets, 'train', batch_size=args.batch_size,  train_type=17)
        
        # To load validation data
        val_loader = load_imputation_price_dataset(args.path_of_imputation_datasets, 'val', batch_size=args.batch_size, train_type=None)
        

        
        
        # 设置设备
        
        # 初始化模型
        model = TimesNet(args).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # 训练模型
        num_epochs = args.epoch
        train_losses, val_losses, val_mae, val_mse, best_model_state = train_imputation_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

        # Save the best model parameters
        model_save_path = './pretrain_args/best_TimesNet_Imputation_model_without_district.pth'
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")
        
        # 使用示例
        save_args(args, './pretrain_args/best_TimesNet_Imputation_model_without_district_args.json')
    
    if args.task == 'test':
        # 模型评估
        # To load test data
        test_loader = load_imputation_price_dataset(args.path_of_imputation_datasets, 'test', batch_size=args.batch_size, train_type=None)
        
        test_loss = 0
        all_inputs = []  # 新增：保存输入部分
        all_pred = []
        all_target = []
        all_nan_mask = []
        all_target_mask = []
        imputation_model = TimesNet(args).to(device)
        imputation_model_path = './pretrain_args/best_TimesNet_Imputation_model_without_district.pth'
        imputation_model.load_state_dict(torch.load(imputation_model_path))
        imputation_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
                all_mask = ~(target_mask + nan_mask)
                outputs = imputation_model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=all_mask.unsqueeze(-1)).squeeze(-1)
                masked_outputs = outputs * target_mask
                masked_targets = targets * target_mask
                loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
                test_loss += loss.item()
                all_inputs.append(x_enc.cpu().numpy())  # 保存输入部分
                all_pred.append(outputs.cpu().numpy())  
                all_target.append(targets.cpu().numpy())
                all_nan_mask.append(nan_mask.cpu().numpy())
                all_target_mask.append(target_mask.cpu().numpy())

        # 转换为numpy数组并堆叠
        all_inputs = np.vstack(all_inputs)      # 输入部分 (batch_size, 48)
        all_pred = np.vstack(all_pred)          # 预测值 (batch_size, 12)
        all_target = np.vstack(all_target)      # 真实值 (batch_size, 12)
        all_nan_mask = np.vstack(all_nan_mask)          # 掩码 (batch_size, 12)
        all_target_mask = np.vstack(all_target_mask)

        full_pred = all_pred[all_target_mask]
        full_target= all_target[all_target_mask]

        # 计算最终指标
        mae = np.mean(np.abs(full_pred - full_target))
        mse = np.mean((full_pred - full_target) ** 2)

        print("-----------逆变换后的指标-----------")
        print(f'Test Restored MSE: {mse}')
        print(f'Test Restored MAE: {mae}')




if __name__ == "__main__":
    main()