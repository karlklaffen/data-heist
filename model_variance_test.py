from preprocessing import X_train, y_train, X_test, y_test, feature_names, address_label_encoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd


y_train_2output = y_train.iloc[:, :2] if hasattr(y_train, 'iloc') else y_train[:, :2]
y_test_2output = y_test.iloc[:, :2] if hasattr(y_test, 'iloc') else y_test[:, :2]

results_3output = []
results_2output = []

print("\nTraining models with different random seeds...\n")

for seed in range(10):
    # 3-Output Model
    model_3 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
    model_3.fit(X_train, y_train)
    pred_3 = model_3.predict(X_test)
    
    # Get Lat/Long R² for 3-output
    lat_r2_3 = r2_score(y_test.iloc[:, 0], pred_3[:, 0])
    long_r2_3 = r2_score(y_test.iloc[:, 1], pred_3[:, 1])
    
    results_3output.append({'lat_r2': lat_r2_3, 'long_r2': long_r2_3, 'avg_r2': (lat_r2_3 + long_r2_3) / 2})
    
    # 2-Output Model
    model_2 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
    model_2.fit(X_train, y_train_2output)
    pred_2 = model_2.predict(X_test)
    
    # Get Lat/Long R² for 2-output
    lat_r2_2 = r2_score(y_test_2output.iloc[:, 0], pred_2[:, 0])
    long_r2_2 = r2_score(y_test_2output.iloc[:, 1], pred_2[:, 1])
    
    results_2output.append({'lat_r2': lat_r2_2, 'long_r2': long_r2_2, 'avg_r2': (lat_r2_2 + long_r2_2) / 2})
    
    print(f"Run {seed+1:2d}  |  3-Output: Lat={lat_r2_3:.6f}, Long={long_r2_3:.6f}  |  2-Output: Lat={lat_r2_2:.6f}, Long={long_r2_2:.6f}")

# Analysis
df_3 = pd.DataFrame(results_3output)
df_2 = pd.DataFrame(results_2output)
\
print("\n3-OUTPUT MODEL (Lat + Long + Address):")
print(f"  Latitude R²:   Mean={df_3['lat_r2'].mean():.6f}, Std={df_3['lat_r2'].std():.6f}, Range={df_3['lat_r2'].min():.6f} to {df_3['lat_r2'].max():.6f}")
print(f"  Longitude R²:  Mean={df_3['long_r2'].mean():.6f}, Std={df_3['long_r2'].std():.6f}, Range={df_3['long_r2'].min():.6f} to {df_3['long_r2'].max():.6f}")
print(f"  Average R²:    Mean={df_3['avg_r2'].mean():.6f}, Std={df_3['avg_r2'].std():.6f}")

print("\n2-OUTPUT MODEL (Lat + Long only):")
print(f"  Latitude R²:   Mean={df_2['lat_r2'].mean():.6f}, Std={df_2['lat_r2'].std():.6f}, Range={df_2['lat_r2'].min():.6f} to {df_2['lat_r2'].max():.6f}")
print(f"  Longitude R²:  Mean={df_2['long_r2'].mean():.6f}, Std={df_2['long_r2'].std():.6f}, Range={df_2['long_r2'].min():.6f} to {df_2['long_r2'].max():.6f}")
print(f"  Average R²:    Mean={df_2['avg_r2'].mean():.6f}, Std={df_2['avg_r2'].std():.6f}")

print("\n" + "-"*80)
print("DIFFERENCE (3-Output minus 2-Output):")
print("-"*80)

lat_diff = df_3['lat_r2'].mean() - df_2['lat_r2'].mean()
long_diff = df_3['long_r2'].mean() - df_2['long_r2'].mean()
avg_diff = df_3['avg_r2'].mean() - df_2['avg_r2'].mean()

lat_diff_pct = (lat_diff / df_2['lat_r2'].mean() * 100) if df_2['lat_r2'].mean() != 0 else 0
long_diff_pct = (long_diff / df_2['long_r2'].mean() * 100) if df_2['long_r2'].mean() != 0 else 0
avg_diff_pct = (avg_diff / df_2['avg_r2'].mean() * 100) if df_2['avg_r2'].mean() != 0 else 0

print(f"  Latitude:   {lat_diff:+.6f} ({lat_diff_pct:+.2f}%)")
print(f"  Longitude:  {long_diff:+.6f} ({long_diff_pct:+.2f}%)")
print(f"  Average:    {avg_diff:+.6f} ({avg_diff_pct:+.2f}%)")

# Statistical test: Is the difference larger than expected noise?
combined_std = np.sqrt(df_3['avg_r2'].std()**2 + df_2['avg_r2'].std()**2)
noise_ratio = abs(avg_diff) / combined_std if combined_std > 0 else 0

print(f"\n  Noise Analysis: Difference is {noise_ratio:.2f}x the combined noise floor")
if noise_ratio > 2:
    print(f"  -> Difference appears SIGNIFICANT (>2x noise)")
elif noise_ratio > 1:
    print(f"  -> Difference is MODERATE (between noise and signal)")
else:
    print(f"  -> Difference appears WITHIN NOISE (likely random variance)")

