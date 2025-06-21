"""
Helper utilities for Forex Chart Pattern Recognition System
"""

import os
import json
import csv
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

def save_results(data: Dict[str, Any], filename: str, format_type: str = 'json') -> bool:
    """Save results to file in specified format"""
    try:
        if format_type.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format_type.lower() == 'csv' and isinstance(data.get('patterns'), list):
            with open(filename, 'w', newline='') as f:
                if data['patterns']:
                    writer = csv.DictWriter(f, fieldnames=data['patterns'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['patterns'])
        elif format_type.lower() == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        return True
    except Exception:
        return False

def load_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load results from file"""
    try:
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                return json.load(f)
        elif filename.endswith('.pickle') or filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception:
        return None

def backup_data(source_dir: str = "data", backup_dir: str = "backups") -> bool:
    """Create backup of data directory"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, backup_path)
            return True
        return False
    except Exception:
        return False

def restore_backup(backup_path: str, target_dir: str = "data") -> bool:
    """Restore data from backup"""
    try:
        if os.path.exists(backup_path):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)
            return True
        return False
    except Exception:
        return False

def validate_data_file(filepath: str) -> Dict[str, Any]:
    """Validate data file and return file info"""
    result = {
        'valid': False,
        'size': 0,
        'records': 0,
        'columns': [],
        'date_range': None,
        'errors': []
    }
    
    try:
        if not os.path.exists(filepath):
            result['errors'].append("File does not exist")
            return result
        
        result['size'] = os.path.getsize(filepath)
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            result['errors'].append("Unsupported file format")
            return result
        
        result['records'] = len(df)
        result['columns'] = df.columns.tolist()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check date range if time column exists
        if 'time' in df.columns or df.index.name == 'time':
            try:
                if 'time' in df.columns:
                    dates = pd.to_datetime(df['time'])
                else:
                    dates = pd.to_datetime(df.index)
                result['date_range'] = {
                    'start': dates.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': dates.max().strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                result['errors'].append(f"Date parsing error: {str(e)}")
        
        result['valid'] = len(result['errors']) == 0
        
    except Exception as e:
        result['errors'].append(f"File reading error: {str(e)}")
    
    return result

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimal places"""
    return f"{value:.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "USD", decimals: int = 4) -> str:
    """Format value as currency"""
    return f"{value:.{decimals}f} {currency}"

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_free': psutil.disk_usage('.').free
    }