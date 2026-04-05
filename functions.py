import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class FileData:
    def __init__(self, filename, folderpath='Datasets/'):
        self.filename = filename
        self.folderpath = folderpath
        self.filepath = os.path.join(folderpath, filename)
    
    @property
    def dataframe(self):
        return pd.read_csv(self.filepath)
    

def _to_numpy(data):
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    return np.asarray(data)

def textToArray(file):
    with open(file, 'r', encoding='utf-8') as file:
        content = file.read()
    tweets = content.splitlines()
    #print(len(tweets))
    return tweets


"""
Calculate regression metrics comparing predictions to expected values.
Input a model_name to print.
"""
def calculate_metrics(y_true, y_pred, model_name=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else np.inf
    
    metrics = {
        'MAE': mae,           # Mean Absolute Error
        'MSE': mse,           # Mean Squared Error
        'RMSE': rmse,         # Root Mean Squared Error
        'R2': r2,             # R² Score (1.0 is perfect)
        'MAPE': mape          # Mean Absolute Percentage Error
    }
    
    # Print to console only accuracy (for console feedback)
    if model_name is not None:
        print(f"  Accuracy: R² = {metrics['R2']:.6f}, MAE = {metrics['MAE']:.6f}")
    
    return metrics


def write_log(log_dict, filename, log_type='model'):
    """
    Write a log dictionary to a formatted txt file in Logs/ directory.
    
    Args:
        log_dict: Dictionary containing log data
        filename: Name of the file (without .txt extension)
        log_type: 'model' or 'analysis' for formatting hints
    """
    log_dir = 'Logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    filepath = os.path.join(log_dir, f"{filename}.txt")
    
    with open(filepath, 'w') as f:
        _write_log_recursive(f, log_dict, indent=0)
    
    print(f"  → Logged to {filepath}")


def _write_log_recursive(file, data, indent=0):
    """Helper function to recursively format and write log data"""
    indent_str = "  " * indent
    next_indent_str = "  " * (indent + 1)
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                file.write(f"{indent_str}{key}:\n")
                _write_log_recursive(file, value, indent + 1)
            else:
                # Format floats nicely
                if isinstance(value, float):
                    file.write(f"{next_indent_str}{key}: {value:.6f}\n")
                else:
                    file.write(f"{next_indent_str}{key}: {value}\n")
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, float):
                        file.write(f"{next_indent_str}{key}: {value:.6f}\n")
                    else:
                        file.write(f"{next_indent_str}{key}: {value}\n")
            else:
                file.write(f"{next_indent_str}[{i}]: {item}\n")
    
    else:
        file.write(f"{indent_str}{data}\n")

'''
tweets = textToArray('Datasets/tweets.txt')
emojis = textToArray('Datasets/emoji.txt')
ConsumerData = FileData('ConsumerData.csv')
FONEData = FileData('FONEData.csv')
USAddressData = FileData('USAddressData.csv')
ZipData = FileData('ZipData.csv')
MigrationData = FileData('OC2025_ZIPMigration.csv')
#print(ConsumerData.shape[0], FONEData.shape[0], USAddressData.shape[0], ZipData.shape[0])
'''
