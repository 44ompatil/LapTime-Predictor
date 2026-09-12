import pandas as pd
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv(r"/home/om/Projects/LapTime-Predictor/data/processed/featEngineeredData.csv")
model_path = Path(__file__).with_name('linear_model.pkl')


X = data.drop(columns=['LapTimeSeconds', 'LapDelta'])
y = data['LapTimeSeconds']

model_data = pd.concat([X, y], axis=1).dropna()
X = model_data.drop(columns=['LapTimeSeconds', 'Sector1TimeSeconds', 'Sector2TimeSeconds', 'Sector3TimeSeconds'])
y = model_data['LapTimeSeconds']

# OHE for categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
	transformers=[
		('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
	],
	remainder='passthrough',
)

model = Pipeline([
	('preprocessor', preprocessor),
	('regressor', LinearRegression()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
with model_path.open('wb') as model_file:
	pickle.dump(model, model_file)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse:.4f}")
print(f"R2 score: {r2:.4f}")
