from flask import Flask
from flask_bootstrap import Bootstrap
import joblib

encoder = joblib.load('mlserve/models/le.joblib')
minmax = joblib.load('mlserve/models/mms.joblib')
model = joblib.load('mlserve/models/lr.joblib')
app = Flask(__name__)
Bootstrap(app)
from mlserve import views