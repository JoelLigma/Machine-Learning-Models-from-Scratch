# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:07:41 2021
@author: Joel

This is just a short project for fun. 
Topic: Implementing Logistic Regression and Gradient Descent from scratch. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================= Quick Data Preprocessing ========================= #
'Source:https://www.kaggle.com/joshmcadams/oranges-vs-grapefruit?select=citrus.csv'
df = pd.read_csv("citrus.csv")
df['citrus'] = df['name'].map({'orange':1,'grapefruit':0}) 
df.drop(columns='name',inplace=True)
# scaling input feature using min-max scaling
scaled_cols = []
for col in df :
    scaled_feature = []
    for i in range(len(df)) :
        scaled_feature += [(df[col][i] - df[col].min()) / (df[col].max() - df[col].min())]
    scaled_cols += [scaled_feature]
    
# create df with scaled features
d = {'diameter': scaled_cols[0],
     'weight': scaled_cols[1],
     'red': scaled_cols[2],
     'green': scaled_cols[3],
     'blue': scaled_cols[4],
     'citrus': list(df["citrus"])} # we dont want to scale y
df2 = pd.DataFrame(d, columns=['diameter','weight','red','green','blue','citrus'])
# shuffle data and reset index
df3 = df2.sample(frac=1,random_state=0)
df3 = df3.reset_index()
df3 = df3.drop(columns="index")
# inset bias term equal to 1
df3.insert(0, 'X0', 1)
# train-test split
X_train = df3.iloc[:, :-1][2000:].values
y_train = df3['citrus'][2000:].values
X_test = df3.iloc[:, :-1][:2000].values
y_test = df3['citrus'][:2000].values

# initialize useful variables for model training
epochs = 5000
lr = 0.01
# get total number of training examples
m_train = len(X_train)
# initialise weights to 0
weights = np.zeros(X_train.shape[1])

# ========================== Model Training ================================ #

def model_accuracy(probabilities,ground_truth) :
    """
    Assigning classes to probabilities and computing the model accuracy.
    """
    accuracy = 1 - sum(abs(np.array([0 if i < 0.5 else 1 for i in np.array(probabilities)])\
                           - np.array(ground_truth))) / len(ground_truth)
    return accuracy

mean_loss_per_epoch = []
for i in range(epochs):
    # fit model and compute mean accuracy for ith epoch
    y_hat = 1/(1+np.exp(-(X_train@weights)))
    mean_train_loss = np.mean(-np.transpose(y_train)*np.log(y_hat)-np.transpose(1-y_train) * \
                            np.log(1-y_hat))
    # keep track of cross entropy loss for ith epoch
    mean_loss_per_epoch += [mean_train_loss]
    # compute model accuracy
    # update weights simultaneously using partial derivative of cross-entropy loss function wrt weights
    weights = weights - lr * (np.transpose(X_train)@(y_hat - y_train)) / m_train
    if i % 1000 == 0 :
        train_accuracy = model_accuracy(y_hat, y_train)
        print(f'Epoch: {i} Training Accuracy: {round(train_accuracy,4)}')
print(f'\nFinal Training Accuracy: {round(train_accuracy,4)}') 
print(f'Final Training Loss: {round(mean_train_loss,4)}\nFinal Parameters: {np.round(weights,3)}') 
print(f"Extrinsic Parameters: Epochs: {epochs}, Learning Rate: {lr}")       
# plotting loss curve
plt.plot(list(range(epochs)),mean_loss_per_epoch,'-b',label='Training Loss')
plt.title('Logistic Regression: Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Training Loss')
plt.show()
        
# measure performance on test set
y_hat = 1/(1+np.exp(-(X_test@weights)))
mean_test_loss = np.mean(-np.transpose(y_test)*np.log(y_hat)-np.transpose(1-y_test) * \
                        np.log(1-y_hat))
test_accuracy = model_accuracy(y_hat, y_test)
print(f'Test Accuracy: {np.round(test_accuracy,4)}')
print(f'Test Loss: {np.round(mean_test_loss,4)}')