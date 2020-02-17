from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib, pandas as pd, numpy as np

# Read our CSV file into a Pandas DataFrame object
df = pd.read_csv('global_power_plant_database.csv')

# Specify the features we would like to use for prediction and drop any rows that are missing data
cols = ['capacity_mw', 'primary_fuel', 'estimated_generation_gwh']
data = df[cols].dropna()

# Create x/y variables for our independent/target variables
x = data.drop('estimated_generation_gwh', axis=1)
y = data.estimated_generation_gwh

# Fit our label encoder and transform our fuel types for training
le = LabelEncoder()
x.primary_fuel = le.fit_transform(x.primary_fuel)
joblib.dump(le, 'le.joblib')

# Fit our MinMax scaler and transform our MW/h capacity for training
scaler = MinMaxScaler()
x.capacity_mw = scaler.fit_transform(np.array(x.capacity_mw).reshape(-1, 1))
joblib.dump(scaler, 'mms.joblib')

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit model and save it
model = LinearRegression()
model.fit(x_train, y_train)
joblib.dump(model, 'lr.joblib')

print(model.score(x_test, y_test))
