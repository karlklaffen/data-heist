import numpy as np
import pandas as pd
from functions import calculate_metrics


def _to_numpy(data):
    """Convert pandas DataFrames to numpy arrays if needed"""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    return np.asarray(data)


def analyze_output_range(y):
    """
    Analyze target variable range to understand prediction difficulty.
    Returns a log dictionary instead of printing.
    """
    y = _to_numpy(y)
    
    lat_min, lat_max = y[:, 0].min(), y[:, 0].max()
    lon_min, lon_max = y[:, 1].min(), y[:, 1].max()
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    if max(lat_range, lon_range) < 0.5:
        interpretation = "NARROW range = Hard problem (predicting fine-grained variation)"
    elif max(lat_range, lon_range) < 5:
        interpretation = "MODERATE range = Acceptable difficulty"
    else:
        interpretation = "WIDE range = Easier problem"
    
    log = {
        'OUTPUT_RANGE_ANALYSIS': {
            'latitude': {
                'min': lat_min,
                'max': lat_max,
                'range': lat_range
            },
            'longitude': {
                'min': lon_min,
                'max': lon_max,
                'range': lon_range
            },
            'interpretation': interpretation
        }
    }
    
    return log


def correlation_analysis(X, y):
    """
    Analyze feature-target correlations to understand signal strength.
    Returns a log dictionary instead of printing.
    """
    X = _to_numpy(X)
    y = _to_numpy(y)
    
    correlations_lat = []
    correlations_lon = []
    
    for i in range(X.shape[1]):
        try:
            corr_lat = np.abs(np.corrcoef(X[:, i], y[:, 0])[0, 1])
            corr_lon = np.abs(np.corrcoef(X[:, i], y[:, 1])[0, 1])
        except:
            corr_lat = np.nan
            corr_lon = np.nan
        correlations_lat.append(corr_lat)
        correlations_lon.append(corr_lon)
    
    max_corr_lat = np.nanmax(correlations_lat) if correlations_lat else 0
    max_corr_lon = np.nanmax(correlations_lon) if correlations_lon else 0
    max_corr_overall = max(max_corr_lat, max_corr_lon)
    
    if max_corr_overall > 0.6:
        signal = "STRONG signal - Complex models justified"
    elif max_corr_overall > 0.3:
        signal = "MODERATE signal - Balance needed"
    else:
        signal = "WEAK signal - Keep models simple"
    
    log = {
        'FEATURE_TARGET_CORRELATION': {
            'max_latitude_correlation': max_corr_lat,
            'max_longitude_correlation': max_corr_lon,
            'max_overall': max_corr_overall,
            'signal_strength': signal
        }
    }
    
    return log


def data_sufficiency_check(X_train, X_test):
    """
    Check if sample-to-feature ratio is healthy.
    Returns a log dictionary instead of printing.
    """
    X_train = _to_numpy(X_train)
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    ratio = n_samples / n_features
    
    if ratio < 10:
        status = "RISKY - High overfitting risk"
    elif ratio < 50:
        status = "ACCEPTABLE - Monitor for overfitting"
    else:
        status = "GOOD - Complex models safe"
    
    log = {
        'FEATURE_SAMPLE_RATIO': {
            'total_samples': n_samples,
            'total_features': n_features,
            'ratio': ratio,
            'status': status
        }
    }
    
    return log


def error_analysis(y_true, y_pred):
    """
    Analyze prediction error distribution.
    Returns a log dictionary instead of printing.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    errors_lat = np.abs(y_true[:, 0] - y_pred[:, 0])
    errors_lon = np.abs(y_true[:, 1] - y_pred[:, 1])
    
    total_mean_error = np.sqrt(errors_lat.mean()**2 + errors_lon.mean()**2)
    
    log = {
        'ERROR_ANALYSIS': {
            'latitude': {
                'mean_error': errors_lat.mean(),
                'std_error': errors_lat.std(),
                'max_error': errors_lat.max()
            },
            'longitude': {
                'mean_error': errors_lon.mean(),
                'std_error': errors_lon.std(),
                'max_error': errors_lon.max()
            },
            'combined_euclidean': total_mean_error
        }
    }
    
    return log


def feature_importance_analysis(model):
    """
    Extract feature importance from tree-based models.
    Returns a log dictionary instead of printing.
    """
    if not hasattr(model.model, 'feature_importances_'):
        log = {
            'FEATURE_IMPORTANCE': {
                'status': 'Feature importance only available for tree models'
            }
        }
        return log
    
    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    top_features = {}
    for i, idx in enumerate(indices):
        if importances[idx] > 0:
            top_features[f'rank_{i+1}'] = {
                'feature_index': int(idx),
                'importance': importances[idx]
            }
    
    log = {
        'FEATURE_IMPORTANCE': {
            'top_features': top_features
        }
    }
    
    return log


def compare_models(models_dict):
    """
    Compare multiple models side-by-side.
    Returns a log dictionary with comparison and inferences.
    """
    comparison_data = []
    for name, metrics in models_dict.items():
        comparison_data.append({
            'Model': name,
            'R2': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE']
        })
    
    comparison = pd.DataFrame(comparison_data).sort_values('R2', ascending=False)
    
    # Convert to dict for logging
    comparison_list = comparison.to_dict('records')
    
    best_r2 = comparison['R2'].iloc[0]
    worst_r2 = comparison['R2'].iloc[-1]
    improvement = ((best_r2 - worst_r2) / abs(worst_r2)) * 100 if worst_r2 != 0 else 0
    
    # Inferences
    inference = ""
    linear_models = comparison[comparison['Model'].str.contains('Linear')]
    tree_models = comparison[comparison['Model'].str.contains('Forest')]
    
    if not linear_models.empty and not tree_models.empty:
        linear_avg = linear_models['R2'].mean()
        tree_avg = tree_models['R2'].mean()
        diff_pct = ((tree_avg - linear_avg) / abs(linear_avg)) * 100 if linear_avg != 0 else 0
        
        if abs(diff_pct) < 5:
            inference = "Signal is LINEAR. Use simpler models (Linear/Ridge)."
        elif tree_avg > linear_avg * 1.2:
            inference = "Signal is NON-LINEAR. Trees/Neural Nets justified."
        else:
            inference = "Marginal difference. Linear models preferred (simpler)."
    
    log = {
        'MODEL_COMPARISON': {
            'ranked_models': comparison_list,
            'best_r2': best_r2,
            'worst_r2': worst_r2,
            'improvement_percent': improvement,
            'inference': inference
        }
    }
    
    return log


def learning_curves(model_class, X_train, y_train, X_test, y_test, fractions=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
    """
    Generate learning curves to see if more data helps.
    Returns a log dictionary instead of printing.
    """
    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train)
    X_test = _to_numpy(X_test)
    y_test = _to_numpy(y_test)
    
    train_sizes = [int(len(X_train) * frac) for frac in fractions]
    results = []
    
    for train_size in train_sizes:
        model = model_class()
        model.train(X_train[:train_size], y_train[:train_size])
        pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, pred)
        results.append({
            'train_size': train_size,
            'percent_data': f"{int(train_size/len(X_train)*100)}%",
            'R2': metrics['R2'],
            'MAE': metrics['MAE']
        })
    
    # Interpretation
    improvement = results[-1]['R2'] - results[0]['R2']
    if improvement < 0.05:
        interpretation = "PLATEAU detected: More data won't significantly help. Focus on feature engineering."
    else:
        interpretation = f"Still IMPROVING: {improvement:.4f} R² gain. More data could help."
    
    log = {
        'LEARNING_CURVES': {
            'curve_data': results,
            'total_improvement': improvement,
            'interpretation': interpretation
        }
    }
    
    return log
