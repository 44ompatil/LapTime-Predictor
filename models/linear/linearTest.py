import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


data_path = Path(__file__).parents[2] / 'data' / 'processed' / 'featEngineeredData.csv'
model_path = Path(__file__).with_name('linear_model.pkl')

if not model_path.exists():
    raise FileNotFoundError(
        f'Model not found at {model_path}. Run linearTrain.py first.'
    )

data = pd.read_csv(data_path)
X = data.drop(
    columns=[
        'LapTimeSeconds',
        'LapDelta',
        'Sector1TimeSeconds',
        'Sector2TimeSeconds',
        'Sector3TimeSeconds',
    ]
)
y = data['LapTimeSeconds']

model_data = pd.concat([X, y], axis=1).dropna()
X = model_data.drop(columns='LapTimeSeconds')
y = model_data['LapTimeSeconds']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with model_path.open('rb') as model_file:
    model = pickle.load(model_file)

y_pred = model.predict(X_test)
absolute_errors = (y_test - y_pred).abs()

print(f'Test samples: {len(y_test)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f} seconds')
print(f'RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.3f} seconds')
print(f'MSE: {mean_squared_error(y_test, y_pred):.3f} seconds^2')
print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Within 1 second: {(absolute_errors <= 1).mean():.2%}')
print(f'Within 2 seconds: {(absolute_errors <= 2).mean():.2%}')