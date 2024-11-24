import logging
import sys
import warnings
from urllib.parse import urlparse

import mlflow as mf
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import os
os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-13-127-189-46.ap-south-1.compute.amazonaws.com:5000/"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    
    #data ingestion - wine quality dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/tests/datasets/winequality-red.csv"
    try:
        data=pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")
    
    train,test = train_test_split(data,test_size=0.2)
    
    train_x = train.drop(["quality"],axis=1)
    test_x = test.drop(["quality"],axis=1)
    
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mf.start_run():
        lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_x,train_y)
        predicted_qualities = lr.predict(test_x)
        rmse,mae,r2=eval_metrics(test_y,predicted_qualities)
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mf.log_param("alpha",alpha)
        mf.log_param("l1_ratio",l1_ratio)
        mf.log_metric("rmse",rmse)
        mf.log_metric("r2",r2)
        mf.log_metric("mae",mae)
        
        #doing setup for the remote server
        remote_uri = "http://ec2-13-127-189-46.ap-south-1.compute.amazonaws.com:5000/"
        mf.set_tracking_uri(remote_uri)
        tracking_url_type_store = urlparse(mf.get_tracking_uri()).scheme
        
        if tracking_url_type_store !="file":
            mf.sklearn.log_model(lr,"model",registered_model_name="ElasticnetWineModel")
        else:
            mf.sklearn.log_model(lr,"model",signature=infer_signature(train_x,test_y))
        
    
    