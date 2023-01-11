import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import warnings
import logging
from pickle import dump
warnings.filterwarnings('ignore')
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger=logging.getLogger("data_processing")


def getData(bucketName, filename):
    # bucketName+'/'+filename
    logger.info("getting data")
    data = pd.read_csv(filename)
    # dataf=data.tail(50).iloc[:,:-1]
    # pd.DataFrame(dataf).to_csv("new_data.csv",index=False)
    return data


def CategoricalDataFunc(data, limit=None):
    # limit of additional categorical values that needs to construct new columns using One hot encoding method.
    if limit == None:
        limit = round(data.shape[0]/100-len(data.columns))
    # we transform the categorical column if unique value less than limited categorical
    logger.info('scanning categorical columns')
    columns = []
    categoricalColumns = []
    for colName, colType in zip(data.columns, data.dtypes):
        numUniqueValue = len(data[colName].unique())
        if numUniqueValue <= limit:
            categoricalColumns.append(colName)
            limit -= numUniqueValue
        elif colType == 'object':
            continue
        else:
            columns.append(colName)
    logger.info('transforming categorical columns using one hot encoding: %s', categoricalColumns)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    dataEncoded = pd.DataFrame(encoder.fit_transform(data[categoricalColumns]))
    dataEncoded.columns = encoder.get_feature_names(categoricalColumns)
    return (pd.concat([dataEncoded, data[columns]], axis=1)), encoder


def MissingDataFixer(data, limit=1):
    drop_index = []
    for i, row in data.iterrows():
        if sum(pd.isna(row)) > limit:
            # removing row if missing value > limit
            drop_index.append(i)
    data = data.drop(drop_index)
    data = data.reset_index(drop=True)
    # If missing values are under the limit for a row, we can do a KNN imputer to fill out missing values
    logger.info("Scaling data using standardSacaler")
    scaler = MinMaxScaler()
    columns = []
    for i, colType in enumerate(data.dtypes):
        if colType == 'float64' or colType == 'int64':
            columns.append(data.columns[i])
    df_scaled = data.copy()
    df_imputed = data.copy()
    # scaling
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    logger.info("filling missing data using KNN imputer")
    # KNN imputing
    imputer = KNNImputer(n_neighbors=45)
    X = imputer.fit_transform(df_scaled[columns])
    df_imputed[columns]=X
    
    return df_imputed, scaler


def SavingData(bucket, oneHotEn, data, scaler):
    logger.info("saving cleaned data and scaler back to bucket %s", bucket)
    dump(oneHotEn, open('oneHotEncoder.pkl', 'wb'))
    np.savetxt("data_cleaned.csv",X=data,delimiter=',')
    dump(scaler, open('scaler.pkl', 'wb'))
    # save to bucket


def main():
    BUCKETNAME = "."
    data = getData(BUCKETNAME, 'data.csv')
    data, scaler = MissingDataFixer(data)
    data, oneHotEn = CategoricalDataFunc(data)
    SavingData(BUCKETNAME, oneHotEn, data, scaler)


if __name__ == "__main__":
    main()
