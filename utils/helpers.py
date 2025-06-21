"""
Helper utilities for the Forex Pattern Recognition System
"""

import json
import csv
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import shutil

def save_results(patterns, filename, format='json'):
    """
    Save pattern detection results to file
    
    Args:
        patterns: List of detected patterns
        filename: Output filename
        format: Output format ('json', 'csv', 'excel')
    """
    try:
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            _save_json_results(patterns, filename)
        elif format.lower() == 'csv':
            _save_csv_results(patterns, filename)
        elif format.lower() in ['excel', 'xlsx']:
            _save_excel_results(patterns, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return True
        
    except Exception as e:
        raise Exception(f"Error saving results: {str(e)}")

def _save_json_results(patterns, filename):
    """Save results as JSON file"""
    # Convert datetime objects and numpy types to serializable formats
    serializable_patterns = []
    
    for pattern in patterns:
        serializable_pattern = {}
        
        for key, value in pattern.items():
            if isinstance(value, datetime):
                serializable_pattern[key] = value.isoformat()
            elif isinstance(value, pd.Timestamp):
                serializable_pattern[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                serializable_pattern[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_pattern[key] = float(value)
            elif isinstance(value, list):
                # Handle nested lists with datetime objects
                serializable_pattern[key] = _serialize_list(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serializable_pattern[key] = _serialize_dict(value)
            else:
                serializable_pattern[key] = value
        
        serializable_patterns.append(serializable_pattern)
    
    # Add metadata
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_patterns': len(patterns),
            'high_confidence_patterns': sum(1 for p in patterns if p.get('confidence', 0) >= 0.8),
            'version': '1.0.0'
        },
        'patterns': serializable_patterns
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def _save_csv_results(patterns, filename):
    """Save results as CSV file"""
    if not patterns:
        # Create empty CSV with headers
        headers = ['pattern_type', 'timestamp', 'confidence', 'signal', 'risk_level']
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return
    
    # Flatten pattern data for CSV
    flattened_data = []
    
    for pattern in patterns:
        row = {
            'pattern_type': pattern.get('type', ''),
            'timestamp': _format_timestamp(pattern.get('timestamp', '')),
            'confidence': pattern.get('confidence', 0),
            'signal': pattern.get('signal', ''),
            'risk_level': pattern.get('risk_level', ''),
            'target_price': pattern.get('target_price', ''),
            'stop_loss': pattern.get('stop_loss', ''),
            'status': pattern.get('status', 'Active')
        }
        
        # Add pattern-specific fields
        if pattern.get('type') == 'Head and Shoulders':
            points = pattern.get('points', [])
            if len(points) >= 3:
                row['left_shoulder_price'] = points[0].get('price', '')
                row['head_price'] = points[1].get('price', '')
                row['right_shoulder_price'] = points[2].get('price', '')
            row['neckline'] = pattern.get('neckline', '')
            
        elif 'Double' in pattern.get('type', ''):
            points = pattern.get('points', [])
            if len(points) >= 2:
                row['first_point_price'] = points[0].get('price', '')
                row['second_point_price'] = points[-1].get('price', '')
            row['support_resistance_level'] = pattern.get('support_level', pattern.get('resistance_level', ''))
            
        elif 'Triangle' in pattern.get('type', ''):
            if 'upper_trendline' in pattern:
                row['upper_slope'] = pattern['upper_trendline'].get('slope', '')
            if 'lower_trendline' in pattern:
                row['lower_slope'] = pattern['lower_trendline'].get('slope', '')
                
        elif 'Support' in pattern.get('type', '') or 'Resistance' in pattern.get('type', ''):
            row['level'] = pattern.get('level', '')
            row['strength'] = pattern.get('strength', '')
            row['touch_count'] = pattern.get('touch_count', '')
        
        flattened_data.append(row)
    
    # Write to CSV
    if flattened_data:
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False, encoding='utf-8')

def _save_excel_results(patterns, filename):
    """Save results as Excel file with multiple sheets"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': ['Total Patterns', 'High Confidence (>80%)', 'Medium Confidence (60-80%)', 
                      'Low Confidence (<60%)', 'Bullish Signals', 'Bearish Signals'],
            'Count': [
                len(patterns),
                sum(1 for p in patterns if p.get('confidence', 0) >= 0.8),
                sum(1 for p in patterns if 0.6 <= p.get('confidence', 0) < 0.8),
                sum(1 for p in patterns if p.get('confidence', 0) < 0.6),
                sum(1 for p in patterns if p.get('signal', '') == 'Bullish'),
                sum(1 for p in patterns if p.get('signal', '') == 'Bearish')
            ]
        }
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed patterns sheet
        if patterns:
            # Create CSV-like flattened data
            _save_csv_results(filename.replace('.xlsx', '.csv'), patterns)
            df = pd.read_csv(filename.replace('.xlsx', '.csv'))
            df.to_excel(writer, sheet_name='Patterns', index=False)
            
            # Clean up temporary CSV
            if os.path.exists(filename.replace('.xlsx', '.csv')):
                os.remove(filename.replace('.xlsx', '.csv'))
        
        # Pattern types sheet
        pattern_types = {}
        for pattern in patterns:
            ptype = pattern.get('type', 'Unknown')
            if ptype not in pattern_types:
                pattern_types[ptype] = {'count': 0, 'avg_confidence': 0, 'total_confidence': 0}
            
            pattern_types[ptype]['count'] += 1
            pattern_types[ptype]['total_confidence'] += pattern.get('confidence', 0)
            pattern_types[ptype]['avg_confidence'] = (
                pattern_types[ptype]['total_confidence'] / pattern_types[ptype]['count']
            )
        
        if pattern_types:
            types_data = []
            for ptype, stats in pattern_types.items():
                types_data.append({
                    'Pattern Type': ptype,
                    'Count': stats['count'],
                    'Average Confidence': f"{stats['avg_confidence']:.1%}",
                    'Percentage': f"{(stats['count'] / len(patterns)) * 100:.1f}%"
                })
            
            pd.DataFrame(types_data).to_excel(writer, sheet_name='Pattern Types', index=False)

def load_results(filename):
    """
    Load pattern detection results from file
    
    Args:
        filename: Input filename
        
    Returns:
        List of patterns or None if error
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.json':
            return _load_json_results(filename)
        elif file_ext == '.csv':
            return _load_csv_results(filename)
        elif file_ext in ['.xlsx', '.xls']:
            return _load_excel_results(filename)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
    except Exception as e:
        raise Exception(f"Error loading results: {str(e)}")

def _load_json_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'patterns' in data:
        return data['patterns']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON format for pattern results")

def _load_csv_results(filename):
    """Load results from CSV file"""
    df = pd.read_csv(filename)
    
    patterns = []
    for _, row in df.iterrows():
        pattern = {
            'type': row.get('pattern_type', ''),
            'timestamp': row.get('timestamp', ''),
            'confidence': float(row.get('confidence', 0)) if pd.notna(row.get('confidence')) else 0,
            'signal': row.get('signal', ''),
            'risk_level': row.get('risk_level', ''),
            'target_price': row.get('target_price', ''),
            'stop_loss': row.get('stop_loss', ''),
            'status': row.get('status', 'Active')
        }
        
        patterns.append(pattern)
    
    return patterns

def _load_excel_results(filename):
    """Load results from Excel file"""
    try:
        # Try to read the Patterns sheet first
        df = pd.read_excel(filename, sheet_name='Patterns')
    except:
        # Fall back to the first sheet
        df = pd.read_excel(filename)
    
    return _load_csv_results(filename.replace('.xlsx', '.csv'))

def _serialize_list(lst):
    """Serialize list with datetime objects"""
    serialized = []
    for item in lst:
        if isinstance(item, datetime):
            serialized.append(item.isoformat())
        elif isinstance(item, pd.Timestamp):
            serialized.append(item.isoformat())
        elif isinstance(item, dict):
            serialized.append(_serialize_dict(item))
        elif isinstance(item, (np.integer, np.floating)):
            serialized.append(float(item))
        else:
            serialized.append(item)
    return serialized

def _serialize_dict(dct):
    """Serialize dictionary with datetime objects"""
    serialized = {}
    for key, value in dct.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, pd.Timestamp):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            serialized[key] = _serialize_list(value)
        elif isinstance(value, dict):
            serialized[key] = _serialize_dict(value)
        elif isinstance(value, (np.integer, np.floating)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized

def _format_timestamp(timestamp):
    """Format timestamp for CSV output"""
    if isinstance(timestamp, (datetime, pd.Timestamp)):
        return timestamp.isoformat()
    elif isinstance(timestamp, str):
        return timestamp
    else:
        return str(timestamp)

def backup_data(source_dir, backup_dir=None, compress=True):
    """
    Create backup of data files
    
    Args:
        source_dir: Source directory to backup
        backup_dir: Backup destination (default: backups/)
        compress: Whether to compress the backup
    """
    try:
        if backup_dir is None:
            backup_dir = "backups"
        
        # Create backup directory
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if compress:
            backup_filename = f"forex_backup_{timestamp}.zip"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
        else:
            backup_filename = f"forex_backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_filename)
            shutil.copytree(source_dir, backup_path)
        
        return backup_path
        
    except Exception as e:
        raise Exception(f"Error creating backup: {str(e)}")

def restore_backup(backup_path, restore_dir):
    """
    Restore data from backup
    
    Args:
        backup_path: Path to backup file/directory
        restore_dir: Directory to restore to
    """
    try:
        if backup_path.endswith('.zip'):
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(restore_dir)
        else:
            shutil.copytree(backup_path, restore_dir)
        
        return True
        
    except Exception as e:
        raise Exception(f"Error restoring backup: {str(e)}")

def clean_old_backups(backup_dir, keep_days=30):
    """
    Clean old backup files
    
    Args:
        backup_dir: Backup directory
        keep_days: Number of days to keep backups
    """
    try:
        if not os.path.exists(backup_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for filename in os.listdir(backup_dir):
            filepath = os.path.join(backup_dir, filename)
            
            if os.path.isfile(filepath):
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_modified < cutoff_date:
                    os.remove(filepath)
                    
    except Exception as e:
        print(f"Error cleaning old backups: {str(e)}")

def validate_data_file(filename):
    """
    Validate forex data file format
    
    Args:
        filename: Path to data file
        
    Returns:
        dict: Validation result
    """
    try:
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(filename)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filename)
        else:
            return {'valid': False, 'error': f'Unsupported file format: {file_ext}'}
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                'valid': False, 
                'error': f'Missing required columns: {missing_columns}'
            }
        
        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {
                    'valid': False,
                    'error': f'Column {col} must be numeric'
                }
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] < 0).any():
                return {
                    'valid': False,
                    'error': f'Column {col} contains negative values'
                }
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            return {
                'valid': False,
                'error': 'High prices must be >= low prices'
            }
        
        return {
            'valid': True,
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "No data"
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def format_number(number, decimal_places=2):
    """Format number for display"""
    try:
        if isinstance(number, (int, float)):
            return f"{number:.{decimal_places}f}"
        else:
            return str(number)
    except:
        return "N/A"

def format_percentage(number, decimal_places=1):
    """Format number as percentage"""
    try:
        if isinstance(number, (int, float)):
            return f"{number * 100:.{decimal_places}f}%"
        else:
            return str(number)
    except:
        return "N/A"

def format_currency(amount, symbol="$", decimal_places=5):
    """Format amount as currency"""
    try:
        if isinstance(amount, (int, float)):
            return f"{symbol}{amount:.{decimal_places}f}"
        else:
            return str(amount)
    except:
        return "N/A"

def get_file_size(filepath):
    """Get file size in human readable format"""
    try:
        size_bytes = os.path.getsize(filepath)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
            
    except:
        return "Unknown"

def ensure_directory(directory):
    """Ensure directory exists"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except:
        return False

def get_system_info():
    """Get system information"""
    try:
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            'disk_total': f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
            'disk_free': f"{psutil.disk_usage('/').free / (1024**3):.2f} GB"
        }
    except ImportError:
        return {'error': 'System info requires psutil package'}
    except Exception as e:
        return {'error': str(e)}

