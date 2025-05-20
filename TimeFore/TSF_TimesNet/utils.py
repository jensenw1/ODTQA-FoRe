import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from model import TimesNet
import argparse

def calculate_metrics(predictions, targets, target_mask):
    valid_pred = predictions[target_mask]
    valid_target = targets[target_mask]
    mae = np.mean(np.abs(valid_pred - valid_target))
    mse = np.mean((valid_pred - valid_target) ** 2)
    return mae, mse

class PriceDataset(Dataset):
    def __init__(self, x, y, target_mask, nan_mask):
        self.x = torch.nan_to_num(torch.FloatTensor(x))
        self.y = torch.nan_to_num(torch.FloatTensor(y))
        self.target_mask = target_mask
        self.nan_mask = nan_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            'target_mask': self.target_mask[idx],
            'nan_mask': self.nan_mask[idx]
        }

def load_imputation_price_dataset(dataset_path, dataset_type, batch_size, train_type):
    """
    Load and prepare price datasets for training, validation, or testing.

    Args:
        dataset_path (str): Path to the directory containing the dataset files
        dataset_type (str): Type of dataset to load ('train', 'val', or 'test')
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 10.

    Returns:
        tuple: (DataLoader, MinMaxScaler) for the specified dataset
    """
    # Load data files
    if dataset_type == 'train':
        data = torch.load(f'{dataset_path}/{dataset_type}_data-{train_type}_nan.pt')
        nan_mask = torch.load(f'{dataset_path}/{dataset_type}_nan_mask-{train_type}_nan.pt').bool()
        target_mask = torch.load(f'{dataset_path}/{dataset_type}_target_mask-{train_type}_nan.pt').bool()
    else:
        data = torch.load(f'{dataset_path}/{dataset_type}_data.pt')
        nan_mask = torch.load(f'{dataset_path}/{dataset_type}_nan_mask.pt').bool()
        target_mask = torch.load(f'{dataset_path}/{dataset_type}_target_mask.pt').bool()
    y = data.clone()
    x = data
    # Apply masks
    x[target_mask] = np.nan
    x[nan_mask] = np.nan
    y[nan_mask] = np.nan
    # Create dataset instance
    dataset = PriceDataset(
        x=x,
        y=y,
        target_mask=target_mask,
        nan_mask=nan_mask
    )
    # Create DataLoader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(dataset_type == 'train'),  # Shuffle only for training
        num_workers=0  # Can be adjusted as needed
    )
    
    return loader

def load_prediction_price_dataset(dataset_path, dataset_type, batch_size=64):
    """
    Load and prepare the price dataset for training, validation, or testing.
    Args:
        dataset_path (str): The directory path containing the dataset files.
        dataset_type (str): The type of dataset to load ('train', 'val', or 'test').
        batch_size (int, optional): The batch size for the DataLoader. Default is 64.
    
    Returns:
        tuple: A tuple containing the specified dataset's (DataLoader, MinMaxScaler).

    """

    data_x = torch.load(f'{dataset_path}/{dataset_type}_data_x.pt')
    data_y = torch.load(f'{dataset_path}/{dataset_type}_data_y.pt')
    x_nan_mask = torch.load(f'{dataset_path}/{dataset_type}_x_nan_mask.pt').bool()
    y_target_mask = torch.load(f'{dataset_path}/{dataset_type}_y_target_mask.pt').bool()
    completed_x = load_and_run_model(data_x, torch.zeros_like(x_nan_mask, dtype=torch.bool), x_nan_mask).cpu()

    dataset = PriceDataset(
        x=completed_x,
        y=data_y,
        target_mask=y_target_mask,
        nan_mask=x_nan_mask
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(dataset_type == 'train'),
        num_workers=0
    )
    return loader


def load_and_run_model(inputs, nan_mask, target_mask, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './pretrain_args/best_TimesNet_Imputation_model_without_district.pth'
    args = load_args('./pretrain_args/best_TimesNet_Imputation_model_without_district_args.json') 
    completion_model = TimesNet(args).to(device)
    completion_model.load_state_dict(torch.load(model_path))
    completion_model.eval()
    with torch.no_grad():
        x_enc = inputs.to(device)
        target_mask = target_mask.to(device)
        nan_mask = nan_mask.to(device)
        mask = ~(target_mask + nan_mask)
        outputs = completion_model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=mask.unsqueeze(-1)).squeeze(-1)
    result = torch.where(mask, x_enc, outputs)
    #print(f'outputs:{outputs}')
    return result

def load_args(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return argparse.Namespace(**data)

def load_price_dataset(dataset_path, dataset_type, batch_size=64):
    """
    Load and prepare the price dataset for training, validation, or testing.

    Args:
        dataset_path (str): The directory path containing the dataset files.
        dataset_type (str): The type of dataset to load ('train', 'val', or 'test').
        batch_size (int, optional): The batch size for the DataLoader. Default is 64.

    Returns:
        tuple: A tuple containing the specified dataset's (DataLoader, MinMaxScaler).
    """
    # 加载数据文件 - 适配新的文件命名方式
    data_x = torch.load(f'{dataset_path}/{dataset_type}_data_x.pt')
    data_y = torch.load(f'{dataset_path}/{dataset_type}_data_y.pt')
    x_nan_mask = torch.load(f'{dataset_path}/{dataset_type}_x_nan_mask.pt').bool()
    y_target_mask = torch.load(f'{dataset_path}/{dataset_type}_y_target_mask.pt').bool()
    completed_x = load_and_run_model(data_x, torch.zeros_like(x_nan_mask, dtype=torch.bool), x_nan_mask).cpu()

    dataset = PriceDataset(
        x=completed_x,
        y=data_y,
        target_mask=y_target_mask,
        nan_mask=x_nan_mask
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(dataset_type == 'train'), 
        num_workers=0 
    )
    return loader



def train_imputation_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_mae = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
            optimizer.zero_grad()
            all_mask = ~(target_mask + nan_mask)
            outputs = model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=all_mask.unsqueeze(-1)).squeeze(-1)
            masked_outputs = outputs * target_mask
            masked_targets = targets * target_mask
            loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        all_val_pred = []
        all_val_target = []
        all_val_target_mask = []
        with torch.no_grad():
            for batch in val_loader:
                x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
                all_mask = ~(target_mask + nan_mask)
                outputs = model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=all_mask.unsqueeze(-1)).squeeze(-1)
                masked_outputs = outputs * target_mask
                masked_targets = targets * target_mask
                loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
                val_loss += loss.item()

                all_val_pred.append(outputs.cpu().numpy())
                all_val_target.append(targets.cpu().numpy())
                all_val_target_mask.append(target_mask.cpu().numpy())

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)


        all_val_pred = np.vstack(all_val_pred)
        all_val_target = np.vstack(all_val_target)
        all_val_target_mask = np.vstack(all_val_target_mask)

        val_mae, val_mse = calculate_metrics(all_val_pred, all_val_target, all_val_target_mask)


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            print(f'Epoch [{epoch+1}/{num_epochs}], New best model with MAE: {val_mae:.4f}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

    print(f'Best validation MAE: {best_val_mae:.4f}')
    return train_losses, val_losses,  val_mae, val_mse, best_model_state


def train_prediction_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_mae = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
            optimizer.zero_grad()
            outputs = model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None).squeeze(-1)
            masked_outputs = outputs * target_mask
            masked_targets = targets * target_mask
            loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        all_val_pred = []
        all_val_target = []
        all_val_target_mask = []
        with torch.no_grad():
            for batch in val_loader:
                x_enc, targets, target_mask, nan_mask = batch['x'].to(device), batch['y'].to(device), batch['target_mask'].to(device), batch['nan_mask'].to(device)
                outputs = model(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None).squeeze(-1)
                masked_outputs = outputs * target_mask
                masked_targets = targets * target_mask
                loss = criterion(masked_outputs, masked_targets) / target_mask.sum().item()
                val_loss += loss.item()

                # 收集验证数据用于计算指标
                all_val_pred.append(outputs.cpu().numpy())
                all_val_target.append(targets.cpu().numpy())
                all_val_target_mask.append(target_mask.cpu().numpy())

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        all_val_pred = np.vstack(all_val_pred)
        all_val_target = np.vstack(all_val_target)
        all_val_target_mask = np.vstack(all_val_target_mask)

        val_mae, val_mse = calculate_metrics(all_val_pred, all_val_target, all_val_target_mask)


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            print(f'Epoch [{epoch+1}/{num_epochs}], New best model with MAE: {val_mae:.4f}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

    print(f'Best validation MAE: {best_val_mae:.4f}')
    return train_losses, val_losses,  val_mae, val_mse, best_model_state


def save_args(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)



def convert_property_data(data_dict):
    """
    Convert the property data dictionary into a list in the specified format.

    Parameters:
        data_dict: A dictionary where the keys are community names and the values are lists of monthly housing price data.
        
    Returns:
        list: A list of tuples in the format [(community_name, month, price), ...].
    """
    result = []
    
    # 定义月份格式
    months = [f"2024年{i}月" for i in range(1, 13)]
    if len(data_dict.keys()) == 1:
        for community, values in data_dict.items():
            values = values[0]
            for i, value in enumerate(values):
                if i < len(months): 
                    # Only add non-zero values to the result list.
                    result.append((months[i], value))
    else:
        # Iterate through each project in the dictionary.
        for community, values in data_dict.items():
            # Create a tuple for each month.
            values = values[0]
            for i, value in enumerate(values):
                if i < len(months):  # Ensure that the index does not exceed the range of the month list.
                    # Only include non-zero values in the result list.
                    result.append((community, months[i], value))
    return result
