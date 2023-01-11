# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sys
from pickle import dump, load
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("hyperparameter_tuning")

# If no user input parameters provided, we will use the followings:
batch_size = [8,16,32] # usually can be 2,4,8,...,(GPU - ParamsSize) / (EstimatedTotalSize ) using summary in torchsummary 
epochs = [10, 50]
optimizer = ['SGD', 'Adam']#, 'Adagrad', Adadelta', 'Adamax', 'Nadam']
activation = ['softmax', 'relu', 'tanh', 'linear']#'sigmoid'
neurons = [5, 10, 20]#,15,25,30
learn_rate = [0.01, 0.1, 0.3]
init_mode = ['uniform', 'normal', 'zero']#, 'lecun_uniform','glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

def load_data(bucket,file_name="data_cleaned.csv"):
    logger.info("loading data from %s",file_name)
    # bucket+'/'+fileName
    dataset = np.loadtxt("data_cleaned.csv", delimiter=",")
    X = dataset[:,:-1]
    Y = dataset[:,-1]
    logger.info("spliting data to training and testing")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    return X_train, X_test, y_train, y_test

def create_model(input_shape=18,activation='linear',init_mode='uniform',neurons=8,optimizer='Adam'):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_shape=(18,),kernel_initializer=init_mode, activation=activation))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
	return model

def append_to_best_group(key,param,best_group,group_size=3):
    if len(best_group)<group_size:
        best_group[key]=param
        return best_group
    curMin=float('inf')
    for curBest in best_group:
        curMin=min(curMin,curBest)
    if curMin<key:
        best_group[key]=param
        del best_group[curMin]
    return best_group
    
    
def hyperparameter_tunning(user_input,X,Y):
    logger.info("hyperparameter tuning process begin")
    model = KerasRegressor(model=create_model, verbose=0)
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer__learning_rate=learn_rate,model__activation=activation,model__neurons=neurons,model__init_mode=init_mode, model__optimizer=optimizer)
    if user_input:
        param_grid=user_input
    logger.info("grid search with user provided input parameters, %s", param_grid)
    logger.info("data splited to training and tuning using 2 fold cross validation")
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
    grid_result = grid.fit(X, Y)
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    best_group=dict()
    for mean, param in zip(means, params):
        best_group=append_to_best_group(mean,param,best_group)
    logger.info("top groups of parmaters: %s", best_group)
    print(best_group)
    return list(best_group.values())

def training_with_cross_validation(best_options_group,X,Y,k=5):
    logger.info("training process begin using the top groups of parameters")
    best_model,best_score=None,0
    for top_option in best_options_group:
        model= KerasRegressor(model=create_model, verbose=0, **top_option)
        # model_result = model.fit(X, Y)
        kf = KFold(n_splits=k, random_state=None)
        folds = kf.split(X, Y)
        scores=[]
        for _, (train_idx, val_idx) in enumerate(folds):
            X_train_cv = X[train_idx]
            y_train_cv = Y[train_idx]
            X_valid_cv = X[val_idx]
            y_valid_cv = Y[val_idx]
            model= KerasRegressor(model=create_model, verbose=0, **top_option)
            model.fit(X_train_cv, y_train_cv, validation_data=(X_valid_cv, y_valid_cv))
            score = np.sqrt(model.score(X_valid_cv, y_valid_cv))
            scores.append(score)
        avg_score=np.mean(scores)
        if avg_score>best_score:
            best_score=avg_score
            best_model=model
    logger.info('best model selected, training accuracy of validation set: %s', best_score)
    return best_model

def save_model(testing_score,best_model_trained,bucket,output_file_name='best_model_trained.pkl'):
    logger.info("saving best model trained to %s", output_file_name)
    #bucket+'/'+file_name     
    dump(best_model_trained,open(output_file_name, 'wb'))     
    dump(testing_score,open("testing_score.txt", 'wb'))   
    with open("testing_score.txt", "w") as outfile:
        outfile.write(testing_score)

def get_testing_score(X_test,y_test,model):
    return model.score(X_test,y_test)

def main():
    user_input=None
    bucket="."
    input_file_name="data_cleaned.csv"
    output_file_name="best_model_trained.pkl"
    if len(sys.argv)>1:
        # input1: bucket
        bucket=sys.argv[1]
    if len(sys.argv)>2:
        input_file_name=sys.argv[2]
    if len(sys.argv)>3:
        # input parameters file name for hyperperameter tunning
        user_input=load(open(sys.argv[3], 'rb'))
    

    # Note: tunning data will be split during tunning so no need to split here
    X_train, X_test, y_train, y_test=load_data(bucket,input_file_name)
    best_options_group=hyperparameter_tunning(user_input,X_train,y_train)
    best_model_trained=training_with_cross_validation(best_options_group,X_train,y_train,k=5)
    testing_score = get_testing_score(X_test,y_test,best_model_trained)
    logger.info("testing score after training is %s",testing_score)
    save_model(testing_score,best_model_trained,bucket,output_file_name)
    


if __name__ == "__main__":
    main()