from regressionmodel import RegressionModel
from preprocessing import X_train, y_train, X_test, y_test
from functions import calculate_metrics, write_log
from dataanalysis import (
    analyze_output_range,
    correlation_analysis,
    data_sufficiency_check,
    error_analysis,
    feature_importance_analysis,
    compare_models,
    learning_curves
)

# ============================================================
# PHASE 1: DATA ANALYSIS
# ============================================================
print("\n" + "="*70)
print("DATA ANALYSIS")
print("="*70)

data_log = {}
data_log.update(analyze_output_range(y_train))
data_log.update(correlation_analysis(X_train, y_train))
data_log.update(data_sufficiency_check(X_train, X_test))

print("✓ Data analysis complete")


# ============================================================
# PHASE 2: TRAIN MODELS (NO SCALER - BEST APPROACH)
# ============================================================
print("\n" + "="*70)
print("MODEL TRAINING - NO SCALER")
print("="*70)

model1 = RegressionModel(model_type='linear', name='Linear (No Scaler)', use_scaler=False)
model2 = RegressionModel(model_type='ridge', name='Ridge (No Scaler)', use_scaler=False)
model3 = RegressionModel(model_type='random_forest', name='Random Forest (No Scaler)', use_scaler=False)

all_models = [model1, model2, model3]

for model in all_models:
    model.train(X_train, y_train)
    print(f"  ✓ {model.name} trained")


# ============================================================
# PHASE 3: MODEL EVALUATION & COMPARISON
# ============================================================
print("\n" + "="*70)
print("MODEL EVALUATION & COMPARISON")
print("="*70)

model_metrics = {}
model_predictions = {}
model_logs = {}

for model in all_models:
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions, model_name=model.name)
    model_metrics[model.name] = metrics
    model_predictions[model.name] = predictions

print("\nModel Comparison:")
comparison_log = compare_models(model_metrics)
for model_name, r2 in sorted([(m, model_metrics[m]['R2']) for m in model_metrics], key=lambda x: x[1], reverse=True):
    print(f"  {model_name}: R² = {r2:.6f}")


# ============================================================
# PHASE 4: ERROR ANALYSIS
# ============================================================
print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

for model in all_models:
    errors = error_analysis(y_test, model_predictions[model.name])
    
    model_log = {
        'model_name': model.name,
        'model_type': model.model_type,
        'use_scaler': model.use_scaler,
        'metrics': model_metrics[model.name]
    }
    model_log.update(errors)
    model_log.update(feature_importance_analysis(model))
    
    model_logs[model.name] = model_log
    print(f"  ✓ {model.name} analyzed")


# ============================================================
# PHASE 5: WRITING LOGS TO FILES
# ============================================================
print("\n" + "="*70)
print("WRITING LOGS TO FILES")
print("="*70)

print("\nData Analysis Log:")
write_log(data_log, 'data_analysis_no_scaler')
comparison_log.update(data_log)
write_log(comparison_log, 'model_comparison_no_scaler')

print("\nModel Logs:")
for model_name, model_log in model_logs.items():
    filename = f"model_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
    write_log(model_log, filename)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nNote: StandardScaler experiments archived in unused/ directory")
print("See unused/observations.txt for performance analysis and recommendations")
