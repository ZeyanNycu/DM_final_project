import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,f1_score
                             
dict_transtoCap = {
    0:'E',
    1:'D',
    2:'C',
    3:'B',
    4:'A'
}

dict_transtoNum = {
    'E':0,
    'D':1,
    'C':2,
    'B':3,
    'A':4
}

class NN:
    def __init__(self,input_num,learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=input_num, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(5,activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    def train(self,X,y,epochs=100,batch_size=128):
        kf = KFold(n_splits=5, shuffle=True)
        f1_list = []
        for train_idx, test_idx in kf.split(y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            y_pred = self.model.predict(X_test,verbose=0)
            y_pred = np.argmax(y_pred,axis=1)
            y_test = np.argmax(y_test,axis=1)
            print(f"Score for fold : f1-score of {f1_score(y_test,y_pred,labels=[0,1,2,3,4],average='macro')}")
            f1_list.append(f1_score(y_test,y_pred,labels=[0,1,2,3,4],average='macro'))
        print('avg f1 : ', np.mean(f1_list))
        
def add_pop_class(data):
    target = 'popularity'
    data.sort_values(target,inplace=True)
    result = np.array_split(data,5)


    for i,data in enumerate(result):
        data['popularity_class'] = dict_transtoCap[i]
        if i > 0:
            result[0] = pd.concat([result[0],result[i]])

    return result[0]

def split_target(data):
    y = data['popularity_class']
    ohc = OneHotEncoder(categories = [['E', 'D', 'C', 'B', 'A']], sparse_output = False)
    y = ohc.fit_transform(y.values.reshape(-1, 1))
    X = data.drop(['popularity','popularity_class'],axis=1)
    X = X.to_numpy()
    return X,y

    
        
    