
## importing the requered dependencies
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import sklearn
import pickle
# importing from sklearn

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

##################################################################################################################################
##################################################################################################################################


def GET_data(log_data:str, new_data:str):
    """this function merges the data sets and removes some redundent columns ['UK Millionaire Maker','DrawNumber']

    Args:
        log_data (str): linke to unstructed historical data 
        new_data (str): linke to unstructed new data  

    Returns:
        _type_: pandas.DataFrame
    """
    # loading the historical log data 
    data_total = pd.read_csv(log_data)
    data_total = data_total.sort_values(by=['Date'],ascending=False, ignore_index= True)
    
    
    # loading the historical New data 
    new_data = pd.read_csv(new_data)
    # converting data standerd form 
    new_data['DrawDate']= pd.to_datetime(new_data['DrawDate'])
    new_data = new_data.sort_values(by=['DrawDate'],ascending=False, ignore_index= True)
    new_data = new_data.rename(columns={"DrawDate": "Date"})
    new_data = new_data.drop(['UK Millionaire Maker','DrawNumber'], axis='columns')

    # merging that two data set with a reperition and in decreasing chronpological order  
    data_TOTAL =  data_total.append(new_data, ignore_index=True)
    data_TOTAL['Date']= pd.to_datetime(data_TOTAL['Date'])
    data_TOTAL = data_TOTAL.sort_values(by=['Date'],ascending=False, ignore_index= True)
    data_TOTAL = data_TOTAL.drop_duplicates()
    return data_TOTAL
###############################################################################################################
###############################################################################################################

def new_row(raw_data, data, i:int ,stack:int):
    """_summary_

    Args:
        raw_data (pandas.DataFrame): structured data 
        data (_type_): _description_
        i (int): _description_
        stack (int): _description_
    """
    aaa = []
    for st in range((stack+1)):
        b = raw_data.loc[i+ st].values.tolist()
        aaa += b
    data.loc[i] = aaa

def data_Stack(raw_data, data, stack):
    """_summary_

    Args:
        raw_data (pandas.DataFrame): structured data 
        data (pandas.DataFrame): _description_
        stack (int): _description_

    Returns:
        _type_:pandas.DataFrame
    """
    for i in range((len(raw_data) - stack)):
        new_row(raw_data, data, i,stack)
    return data

def Data_formed_and_stacked(data_TOTAL, stack:int):
    """this will add the previous draw n day to the individual rows, where n is equal to stack(ie: lookback)

    Args:
        data_TOTAL (pandas.DataFrame): imported historical data 
        stack (int): etting the how far the ml will be able to look back at historical data  

    Returns:
        _type_:pandas.DataFrame
    """
    columns_name3 =['Date','Ball_1','Ball_2','Ball_3','Ball_4','Ball_5','Lucky_Star_1','Lucky_Star_2']
    for index in range(1,(stack+1)):
        lcls = locals()
        exec("lol=['Date_{name1}','Ball_1_{name1}','Ball_2_{name1}','Ball_3_{name1}','Ball_4_{name1}','Ball_5_{name1}','Lucky_Star_1_{name1}','Lucky_Star_2_{name1}']".format(name1=index), globals(),lcls )
        lol = lcls["lol"]
        columns_name3 = columns_name3 + lol
    data_Farme = pd.DataFrame(columns=columns_name3)
    return (data_Stack(data_TOTAL ,data_Farme ,stack)) 


def form_dateDIFF(data):
    """encoding date columns such that the oldest draw date is set to zero

    Args:
        data (pandas.DataFrame): starcked historical data 

    Returns:
        _type_: pandas.DataFrame
    """
    ddd1 = pd.DataFrame()
    for name in data.columns:
        if 'Date_' in name :
            ddd1[name] =  data['Date'] - data[name]
        else:
            pass
    # re-setting the data values 
    ddd1['Date'] =  data['Date'] - data.loc[len(data)-1]['Date']
    #
    for col in ddd1.columns:
        ddd1[str(col)] = (ddd1[str(col)]).dt.days
    #
    for col in ddd1.columns:
        data[str(col)] = ddd1[str(col)]
    return (data)


##################################################################################################################################
##################################################################################################################################
def to_int(data):
    data_ceil = data.copy()
    for count_list, list_ in enumerate(data):
        data_ceil[count_list] = np.rint(list_)#np.ceil(list_)# round to ceil
    return (data_ceil)# (data_trunc ,data_rounded, data_ceil, data_floor)
##################################################################################################################################
##################################################################################################################################

## ML code 
def SPLIT_1(data, b_val):
    """ spliting the data to data_x and data_y. 
        each row in the data_y will be a list of b_val rows
        the selected columns will be removed from data_x

    Args:
        data (pandas.DataFrame): structure historical data 
        b_val (int): numbers of rows

    Returns:
        _type_: pandas.DataFrame
    """

    
    data_y = []
    for i in range(len(data)):
        aaa = []
        aaa.append(data.loc[i][b_val])
        data_y.append(aaa)
    drop_list = ['Ball_1', 'Ball_2','Ball_3', 'Ball_4', 'Ball_5', 'Lucky_Star_1','Lucky_Star_2']                     
    data_x = data.drop(columns=drop_list[b_val-1:])
    return data_x, data_y
##################################################################################################################################
##################################################################################################################################

## ML code 
def SPLIT_2(data, b_val):
    """ Spliting the data to data_x and data_y. Both are pandas DatatFrame.
        Eeach row in the data_y will be a list of b_val rows 
        the data_x  = data

    Args:
        data (pandas.DataFrame): structure historical data 
        b_val (int): numbers of rows

    Returns:
         _type_: pandas.DataFrame
    """
    

    drop_list = ['Ball_1', 'Ball_2','Ball_3', 'Ball_4', 'Ball_5', 'Lucky_Star_1','Lucky_Star_2']
    data_y = pd.DataFrame()
    for val in drop_list:
        data_y[val] = data[val]
    data_x = data #.drop(columns=drop_list)
    return data_x, data_y

##################################################################################################################################
##################################################################################################################################
def SPLIT_3(DaTa, D_type=1):
    """allows us to split the data in two different ways
        D_type=0: saves the firs 10% of the data for prediction and the rest to training the models.
        D_type=1: saves the some data at random for prediction and the rest to training the models.

    Args:
        DaTa (pandas.DataFrame): structure historical data 
        D_type (int, optional): this is logic input. Defaults to 1.

    Returns:
        _type_: pandas.DataFrame
    """
    if D_type == 0: 
        ## reserving the first 10 percent tof the data 
        for_test = int(len(DaTa)*0.10) #10
        Data_for_prediction = DaTa.head(for_test)

        # droping the data 
        val = list()
        val = list(Data_for_prediction.index.values)

        DaTa = DaTa.drop(val)
        DaTa.sort_index(inplace=True) 

    else:
        ## reserving some data at random 
        data_top_row = DaTa.head(3)

        for_test = int((len(DaTa)*0.10) - 3) #99
        Data_for_prediction_prime = DaTa.sample(n=for_test, random_state=1, replace=True)

        frames = [data_top_row, Data_for_prediction_prime ]
        Data_for_prediction = pd.concat(frames)
        
        val = list(Data_for_prediction.index.values)

        DaTa = DaTa.drop(val)
        DaTa.sort_index(inplace=True) 
    return DaTa , Data_for_prediction

##################################################################################################################################
##################################################################################################################################

def creat_models(data, data_for_pred, models):
    """this function used the training set and it is used on all ml models in the models dictionary, 
    and keeps the best modles best on  f1scores of said model.

    Args:
        data (pandas.DataFrame): structure historical data resered for training 
        data_for_pred (pandas.DataFrame): structure historical data resered for testing 
        models (structure historical data resered for training ): dictionary of name and ml models.

    Returns:
        _type_: tuple, dict_model, dict_score, dict_y_pred, y_for_pred 
    """
    name, model = models 
    print (name)
    dict_model = {}
    dict_score = {}
    dict_y_pred = {}
    p_list = ['Ball_1', 'Ball_2','Ball_3', 'Ball_4', 'Ball_5', 'Lucky_Star_1','Lucky_Star_2']
    
    # split the x and y both panda DataFrame 
    X, y = SPLIT_2(data = data, b_val = len(p_list))#b_val = 7)
    X_for_pred, y_for_pred = SPLIT_2(data = data_for_pred, b_val = len(p_list))#b_val = 7)

    #split of data train =0.9 and test = 0.1   
    X_train_prime, X_test_prime, y_train_prime , y_test_prime = train_test_split(X, y , test_size=0.25, shuffle=False)
    #
    #
    ## the main loop
    for TT, TT_value in enumerate(p_list):
        print(TT_value)

        ## remove excess columns from X_train, X_test and X_pred
        X_train = X_train_prime.drop(columns=p_list[TT:]) # this is the input data to train the model
        cols = [c for c in X_train.columns if TT_value in c]
        X_train = X_train[cols]
  
        
        X_test  = X_test_prime.drop(columns=p_list[TT:]) # this is the input data to test the model
        cols = [c for c in X_test.columns if TT_value in c]
        X_test = X_test[cols]
        
        X_pred = X_for_pred.drop(columns=p_list[TT:]) # this is the input data to predict the model
        cols = [c for c in X_pred.columns if TT_value in c]
        X_pred = X_pred[cols]
        
        ## setting column from y_train and y_test
        y_train = y_train_prime[TT_value] # this is the target data to train the model
        y_test = y_test_prime[TT_value] # this is the target data to test the model
        y_true = y_for_pred[TT_value] # this is the target data to test the model
        ## generating the model 
        model.fit(X_train, y_train)
        ## saving the data 
        dict_model[TT_value] = model
        
        y_pred = model.predict(X_pred) # to_int(model.predict(X_test))
        dict_y_pred[TT_value] = y_pred
        dict_score[TT_value] = accuracy_score(y_true, y_pred)
        if 'Ball_1' ==  TT_value:
            path = 'model_linear/'+ name
            if not os.path.exists(path):
                os.makedirs(path)
            path_to_file = 'model_linear/'+ name +'/'+'model_Ball_1.pkl'
            modelfile = open(path_to_file,'wb+')
            pickle.dump(model, modelfile)
            modelfile.close()
            
            with open('model_linear/temp_testing_model_Ball_1.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Ball_1.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Ball_2' ==  TT_value:
            with open('model_linear/temp_testing_model_Ball_2.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Ball_2.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Ball_3' ==  TT_value:
            with open('model_linear/temp_testing_model_Ball_3.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Ball_3.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Ball_4' ==  TT_value:
            with open('model_linear/temp_testing_model_Ball_4.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Ball_4.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Ball_5' ==  TT_value:
            with open('model_linear/temp_testing_model_Ball_5.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Ball_5.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Lucky_Star_1' ==  TT_value:
            with open('model_linear/temp_testing_model_Lucky_Star_1.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Lucky_Star_1.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
                
        if 'Lucky_Star_2' ==  TT_value:
            with open('model_linear/temp_model_Lucky_Star_2.pkl', 'wb') as file:  
                pickle.dump(model, file)
            with open('model_linear/X_train_Lucky_Star_1.pkl', 'wb') as file:  
                pickle.dump(X_train, file)
        
    return (dict_model, dict_score, dict_y_pred, y_for_pred)


##################################################################################################################################
##################################################################################################################################
def new_data_for_pred(log_data:str, new_data:str, stack:int = 15):
    """_summary_

    Args:
        log_data (str): _description_
        new_data (str): _description_
        stack (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    ## geting the last date recorded day the of draw
    data_log = pd.read_csv(log_data, index_col=False )
    last_date = data_log.iloc[0]['Date']
    
    #combined data
    data_TOTAL = GET_data(log_data=log_data, new_data=new_data)
    #stacked data
    TOTAL_DATA = Data_formed_and_stacked(data_TOTAL = data_TOTAL, stack = stack)
    #remove the old data
    index_val = TOTAL_DATA[TOTAL_DATA['Date']==last_date].index.values
    new_new_data = TOTAL_DATA.head(index_val+1)
    
    #encod data
    new_new_data = form_dateDIFF(new_new_data)
    new_new_data[new_new_data.columns] = new_new_data[new_new_data.columns].applymap(np.int64)
    
    return new_new_data
##################################################################################################################################
##################################################################################################################################

def get_pred(data, data_for_pred, list_model):
    """_summary_

    Args:
        data (_type_): _description_
        data_for_pred (_type_): _description_
        list_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    ulrimate_models = {'Ball_1': 0, 'Ball_2': 0,'Ball_3': 0, 'Ball_4': 0, 'Ball_5': 0, 'Lucky_Star_1': 0,'Lucky_Star_2': 0}
    ulrimate_y_pred = {'Ball_1': 0, 'Ball_2': 0,'Ball_3': 0, 'Ball_4': 0, 'Ball_5': 0, 'Lucky_Star_1': 0,'Lucky_Star_2': 0}
    ulrimate_score = {'Ball_1': 0, 'Ball_2': 0,'Ball_3': 0, 'Ball_4': 0, 'Ball_5': 0, 'Lucky_Star_1': 0,'Lucky_Star_2': 0}
    #
    ulrimate_y_pred_all = {'Ball_1': [0], 'Ball_2': [0],'Ball_3': [0], 'Ball_4': [0], 'Ball_5': [0], 'Lucky_Star_1': [0],'Lucky_Star_2': [0]}
    ulrimate_y_pred_all = pd.DataFrame.from_dict(data = ulrimate_y_pred_all)
    #
    p_list = ['Ball_1', 'Ball_2','Ball_3', 'Ball_4', 'Ball_5', 'Lucky_Star_1','Lucky_Star_2']
    #
    for hmm, hmm_val in enumerate(list_model):
        dict_model, dict_score, dict_y_pred ,y_for_pred= creat_models(data = data, data_for_pred = data_for_pred ,models = hmm_val)
        
        dict_y_pred = pd.DataFrame.from_dict(dict_y_pred)
        ulrimate_y_pred_all.append(dict_y_pred.head(1))
        
        for ymm_val in p_list:
            if ulrimate_score[ymm_val] < dict_score[ymm_val]:
                ulrimate_score[ymm_val] = dict_score[ymm_val]
                ulrimate_models[ymm_val] = dict_model[ymm_val]
                ulrimate_y_pred[ymm_val] = dict_y_pred[ymm_val]
                
                if 'Ball_1' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Ball_1.pkl",'rb')
                    model_Ball_1 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Ball_1.pkl', 'wb') as file:  
                        pickle.dump(model_Ball_1, file)
                    
                if 'Ball_2' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Ball_2.pkl",'rb')
                    model_Ball_2 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Ball_2.pkl', 'wb') as file:  
                        pickle.dump(model_Ball_2, file)
                    
                if 'Ball_3' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Ball_3.pkl",'rb')
                    model_Ball_3 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Ball_3.pkl', 'wb') as file:  
                        pickle.dump(model_Ball_3, file)
                    
                if 'Ball_4' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Ball_4.pkl",'rb')
                    model_Ball_4 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Ball_4.pkl', 'wb') as file:  
                        pickle.dump(model_Ball_4, file)
                    
                if 'Ball_5' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Ball_5.pkl",'rb')
                    model_Ball_5 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Ball_5.pkl', 'wb') as file:  
                        pickle.dump(model_Ball_5, file)
                    
                if 'Lucky_Star_1' ==  ymm_val:
                    infile = open("model_linear/temp_testing_model_Lucky_Star_1.pkl",'rb')
                    model_Lucky_Star_1 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Lucky_Star_1.pkl', 'wb') as file:  
                        pickle.dump(model_Lucky_Star_1, file)
                    
                if 'Lucky_Star_2' ==  ymm_val:
                    infile = open("model_linear/temp_model_Lucky_Star_2.pkl",'rb')
                    model_Lucky_Star_2 = pickle.load(infile)
                    infile.close()
                    with open('model_linear/holding_real_model_Lucky_Star_2.pkl', 'wb') as file:  
                        pickle.dump(model_Lucky_Star_2, file)
    
    return (ulrimate_models, ulrimate_score, ulrimate_y_pred,y_for_pred, ulrimate_y_pred_all)


def val_difference_between_pandas(y_for_pred, y_pred):
    """_summary_

    Args:
        y_for_pred (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_for_pred = y_for_pred.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    df_diff = y_for_pred.copy()
    for col_name in list(y_pred.columns):
        df_diff[col_name] = y_for_pred[col_name] - y_pred[col_name]
    
    # drop first colum
    df_diff = df_diff.iloc[1: , :]
    ave_list = []
    for col_name in list(y_pred.columns):
        ave_list.append( df_diff[col_name] .mean())
    
    lol = y_pred.loc[0, :].values.tolist()
    ave_val = [sum(x) for x in zip(lol, ave_list)]
    return ave_val





if __name__ == '__main__':
    pass



























