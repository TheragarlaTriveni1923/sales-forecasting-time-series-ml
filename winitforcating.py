import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


TARGET_RMSE = 10
TARGET_MAPE = 0.15


TRAIN_FILE = r'C:\Users\a\Downloads\train.csv'
TEST_FILE = r'C:\Users\a\Downloads\test.csv'


try:
    train_data = pd.read_csv(TRAIN_FILE, parse_dates=['date'])
    test_data = pd.read_csv(TEST_FILE, parse_dates=['date'])
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")


df_ts = train_data.set_index('date').sort_index()
sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))
df_ts['sales'].plot(title='Sales Over Time', color='blue')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
sns.boxplot(data=train_data, x='store', y='sales', ax=axes[0], palette="pastel")
sns.boxplot(data=train_data, x='item', y='sales', ax=axes[1], palette="pastel")
axes[0].set_title('Sales Distribution by Store')
axes[1].set_title('Sales Distribution by Item')
plt.tight_layout()
plt.show()


print("1. Trend: Summarize if sales show an increasing, decreasing, or cyclical pattern as observed.")
print("2. Seasonality: Summarize visible seasonal effects, e.g., higher sales during holidays/months.")
print("3. Outliers: Note any apparent data anomalies or sudden spikes/drops.")


def create_features(df_input, historical_df):
    df = df_input.copy()
    full_data = pd.concat([historical_df, df]).sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    full_data['year'] = full_data['date'].dt.year
    full_data['month'] = full_data['date'].dt.month
    full_data['day_of_week'] = full_data['date'].dt.dayofweek
    full_data['day_of_month'] = full_data['date'].dt.day
    full_data['day_of_year'] = full_data['date'].dt.dayofyear
    full_data['week_of_year'] = full_data['date'].dt.isocalendar().week.astype(int)
    for lag in [1, 7, 365]:
        full_data[f'sales_lag_{lag}'] = full_data.groupby(['store', 'item'])['sales'].shift(lag)
    for window in [7, 30]:
        full_data[f'sales_roll_mean_{window}'] = full_data.groupby(['store', 'item'])['sales'].rolling(window=window).mean().reset_index(level=[0,1], drop=True)
    full_data = pd.get_dummies(full_data, columns=['store', 'item'], prefix=['store', 'item'])
    df_with_lags = full_data.iloc[-len(df):].copy()
    return df_with_lags

df_train_feat = create_features(train_data, train_data.copy())
df_train_feat_clean = df_train_feat.dropna(subset=['sales_lag_365']).copy()
FEATURES = [col for col in df_train_feat_clean.columns if col not in ['date', 'sales', 'id']]
TARGET = 'sales'
X_train = df_train_feat_clean[FEATURES]
y_train = df_train_feat_clean[TARGET]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(X_train.head())
print(y_train.head())

N_VALIDATION_DAYS = 90
x_val = X_train.iloc[-N_VALIDATION_DAYS:]
y_val = y_train.iloc[-N_VALIDATION_DAYS:]
x_train_final = X_train.iloc[:-N_VALIDATION_DAYS]
y_train_final = y_train.iloc[:-N_VALIDATION_DAYS]
print(f"Training data (for models) shape: {x_train_final.shape}, {y_train_final.shape}")

def evaluate_model(y_true, y_pred, model_name='Model'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f} (target: {TARGET_RMSE}), MAPE: {mape:.4f} (target: {TARGET_MAPE})")
    return rmse, mape


print("\nMODEL: Linear Regression (Baseline)")
lr_model = LinearRegression()
lr_model.fit(x_train_final.fillna(0), y_train_final)
y_lr_pred = lr_model.predict(x_val.fillna(0))
y_lr_pred = np.maximum(0, y_lr_pred)  # Sales can't be negative
rmse_lr, mape_lr = evaluate_model(y_val, y_lr_pred, "Linear Regression")
best_model = lr_model
chosen_rmse, chosen_mape = rmse_lr, mape_lr


print("\nMODEL: Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
rf_model.fit(x_train_final.fillna(0), y_train_final)
y_rf_pred = rf_model.predict(x_val.fillna(0))
y_rf_pred = np.maximum(0, y_rf_pred)
rmse_rf, mape_rf = evaluate_model(y_val, y_rf_pred, "Random Forest")


if rmse_rf < chosen_rmse and mape_rf < chosen_mape:
    best_model = rf_model
    chosen_rmse = rmse_rf
    chosen_mape = mape_rf
    print("Random Forest chosen as BEST_MODEL")
else:
    print("Linear Regression remains as BEST_MODEL")


val_plot_df = pd.DataFrame({
    'ACTUAL Sales': y_val.values,
    'Predicted Sales': np.maximum(0, best_model.predict(x_val.fillna(0)))
})
val_plot_df_sample = val_plot_df.head(100).reset_index(drop=True)
plt.figure(figsize=(12, 5))
sns.lineplot(data=val_plot_df_sample, x=val_plot_df_sample.index, y='ACTUAL Sales', label='Actual Sales', marker='o')
sns.lineplot(data=val_plot_df_sample, x=val_plot_df_sample.index, y='Predicted Sales', label='Predicted Sales')
plt.xlabel('Time Step (Sample)')
plt.ylabel('Sales')
plt.legend()
plt.show()

if chosen_rmse < TARGET_RMSE and chosen_mape < TARGET_MAPE:
    print(f"Chosen model ({type(best_model).__name__}) meets success targets on validation set. Proceeding to final forecast.")
else:
    print(f"Chosen model ({type(best_model).__name__}) failed to meet targets. Consider hyperparam tuning or more features.")

print("\nTraining best model on full data for final forecast...")
best_model.fit(X_train.fillna(0), y_train)
df_test_feat = create_features(test_data, train_data)
X_test = df_test_feat[FEATURES]
test_preds = best_model.predict(X_test.fillna(0))
test_preds = np.maximum(0, test_preds)


submission = test_data.copy()
submission['sales'] = test_preds
submission[['id', 'sales']].to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created.")

