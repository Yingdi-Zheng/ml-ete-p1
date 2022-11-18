import pkg_resources
import importlib
importlib.reload(pkg_resources)
import tensorflow as tf
import tensorflow_data_validation as tfdv
import os
import tempfile, urllib, zipfile



# def getData():
#     BASE_DIR = tempfile.mkdtemp()
#     DATA_DIR = os.path.join(BASE_DIR, 'data')
#     OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')
#     TRAIN_DATA = os.path.join(DATA_DIR, 'train', 'data.csv')
#     EVAL_DATA = os.path.join(DATA_DIR, 'eval', 'data.csv')
#     SERVING_DATA = os.path.join(DATA_DIR, 'serving', 'data.csv')

#     # Download the zip file from GCP and unzip it
#     zip, headers = urllib.request.urlretrieve('https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/chicago_data.zip')
#     zipfile.ZipFile(zip).extractall(BASE_DIR)
#     zipfile.ZipFile(zip).close()
#     train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)

def DataProcessing():
    print("dataprocessing") 

def main():
    DataProcessing()

if __name__ == "__main__":
    main()