from pickle import load
import logging 
import pandas as pd
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("prediction")

def load_info(data_file_name,scaler_file_name,encoder_file_name,model_file_name,bucket_name):
    return pd.read_csv(data_file_name),load(open(scaler_file_name, 'rb')),load(open(encoder_file_name, 'rb')),load(open(model_file_name, 'rb'))

def regenerate_data_as_input(data,scaler,encoder):
    logger.info("Scaling data using stored sacaler")
    columns = []
    for i, colType in enumerate(data.dtypes):
        if colType == 'float64' or colType == 'int64':
            columns.append(data.columns[i])
    df_scaled = data.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    categoricalColumns=encoder.get_feature_names_out()
    dataEncoded = pd.DataFrame(encoder.transform(data[categoricalColumns]))
    dataEncoded.columns = encoder.get_feature_names(categoricalColumns)
    return (pd.concat([dataEncoded, data[columns]], axis=1)), encoder

def predict(new_data_input,model,scaler):
    result = model.predict(new_data_input)
    scaler.inverse_transform(result)
    return pd.DataFrame(scaler.inverse_transform(result))

def save_result(result,bucket_name,file_name):
    #bucket_name+'/'+file_name 
    result.to_csv(file_name,index=False)

def main():
    bucket_name="."
    data_file_name="new_data.csv"
    scaler_file_name="scaler.pkl"
    encoder_file_name="oneHotEncoder.pkl"
    model_file_name="best_model_trained.pkl"
    new_records,scaler,encoder,model=load_info(data_file_name,scaler_file_name,encoder_file_name,model_file_name,bucket_name)
    new_data_input =regenerate_data_as_input(new_records,scaler,encoder)
    result=predict(new_data_input,encoder,model)
    save_result(result,bucket_name,file_name='output.csv')

    
if __name__ == "__main__":
    main()