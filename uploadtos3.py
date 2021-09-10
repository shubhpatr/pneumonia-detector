# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:20:38 2021

@author: mohap
"""


import s3fs
import zipfile
import tempfile
import numpy as np
from tensorflow import keras
from pathlib import Path
import logging
import os

AWS_ACCESS_KEY="AKIA532TKVC54GUI4BES"
AWS_SECRET_KEY="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"
BUCKET_NAME="cnn-model"
tempdir = './models'
tempdir2 = './extracted'
model_name = 'cnn model'


def get_s3fs():
  return s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)


def zipdir(path, ziph):
  # Zipfile hook to zip up model folders
  length = len(path) # Doing this to get rid of parent folders
  for root, dirs, files in os.walk(path):
    folder = root[length:] # We don't need parent folders! Why in the world does zipfile zip the whole tree??
    for file in files:
      ziph.write(os.path.join(root, file), os.path.join(folder, file))

            
def s3_save_keras_model():
  # with tempfile.TemporaryDirectory() as tempdir:
  #   model.save(f"{tempdir}/{model_name}")
    # Zip it up first
    zipf = zipfile.ZipFile(f"./zipfile/{model_name}.zip", "w", zipfile.ZIP_STORED)
    zipdir(f"{tempdir}", zipf)
    zipf.close()
    s3fs = get_s3fs()
    s3fs.put(f"./zipfile/{model_name}.zip", f"{BUCKET_NAME}/{model_name}.zip")
    logging.info(f"Saved zipped model at path s3://{BUCKET_NAME}/{model_name}.zip")
 

def s3_get_keras_model(model_name: str) -> keras.Model:
  with tempfile.TemporaryDirectory() as tempdir:
    s3fs = get_s3fs()
    # Fetch and save the zip file to the temporary directory
    s3fs.get(f"{BUCKET_NAME}/{model_name}.zip", f"{tempdir2}/{model_name}.zip")
    # Extract the model zip file within the temporary directory
    with zipfile.ZipFile(f"{tempdir2}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir2}/{model_name}")
        
    # Load the keras model from the temporary directory
    return keras.models.load_model(f"{tempdir2}/{model_name}//{model_name}")
  

# Save the model to S3
# s3_save_keras_model() 

# Load the model from S3
loaded_model = s3_get_keras_model(model_name)