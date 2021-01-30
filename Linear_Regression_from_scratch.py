# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 23:12:55 2021
@author: Joel

This is just a short project for fun and for teaching myself vectorized implementation. 

Topic: Implementing Linear Regression with a vectorizied implementation
of Gradient Descent from scratch.

1. Univariate first order 
2. Univariate second order
3. Multivariate first order - For Loop vs. Vectorized 
"""
import numpy as np
import math,time
import pandas as pd
import matplotlib.pyplot as plt

# toggle on/off
uni_first_order = True
uni_second_order = True
multi_first_order = True
loop = True
vectorized = True

# ======================= Quick Data Preprocessing ========================= #
'Source: https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict.csv'
df = pd.read_csv("Admission_predict.csv")
df.drop(columns="Serial No.", inplace=True)
# scaling input feature using min-max scaling
scaled_cols = []
for col in df :
    scaled_feature = []
    for i in range(len(df)) :
        scaled_feature += [(df[col][i] - df[col].min()) / (df[col].max() - df[col].min())]
    scaled_cols += [scaled_feature]
# create df using all features
d = {'gre_score': scaled_cols[0],
     'TOEFL Score': scaled_cols[1],
     'University Rating': scaled_cols[2],
     'SOP': scaled_cols[3],
     'LOR': scaled_cols[4],
     'CGPA': scaled_cols[5],
     'Research': scaled_cols[6],
     'chance_of_admit': list(df["Chance of Admit "])} # we dont want to scale y
df2 = pd.DataFrame(d, columns=['gre_score','TOEFL Score','University Rating',
                               'SOP','LOR','CGPA','Research','chance_of_admit'])
# create df using only gre score and the response feature chance of admit
# shuffle data and reset index
df3 = df2.sample(frac=1,random_state=0)
df3 = df3.reset_index()
df3 = df3.drop(columns="index")
# inset bias term equal to 1
df3.insert(0, 'X0', 1)
# Train-Test Split
# get training feature values for vectorized implementation
X_train = df3.iloc[:, 0:2][100:].values
y_train = df3['chance_of_admit'][100:].values
# get test feature values for vectorized implementation
X_test = df3.iloc[:, 0:2][:100].values
y_test = df3['chance_of_admit'][:100].values

# initialize useful variables for model training
epochs = 10000
lr = 0.001
# get total number of training examples
m_train = len(X_train)
m_test = len(X_test)

# ================ First-Order Univariate Linear Regression ================ #
if uni_first_order :
    
    # initialise weights 
    weights = np.zeros(X_train.shape[1])
    # Training Loop
    RMSE_values = []
    epoch_values = []
    for i in range(epochs):
        epoch_values += [i]
        # fit model and compute cost for ith epoch
        RMSE1 = np.sqrt((1/(2*m_train)) * np.transpose(((X_train@weights) - y_train))@\
                       ((X_train@weights) - y_train))
        # keep track of loss for ith epoch
        RMSE_values += [RMSE1]
        # update weights (works simultaneously)
        weights = weights - lr * ((1/m_train) * np.transpose(X_train)@((X_train@weights) - y_train)) # partial derivative of L2-norm wrt theta1
        if i % 1000 == 0 :
            print(f'Epoch: {i} Training RMSE: {round(RMSE1,3)}')
    print(f'\nFinal Training RMSE1: {round(RMSE1,3)}\nFinal Parameters: {np.round(weights,3)}') 
    print(f"Extrinsic Parameters: Epochs: {epochs}, Learning Rate: {lr}")
    
    # plot loss curve
    plt.plot(epoch_values,RMSE_values,'-b',label='RMSE')
    plt.title('Uni 1st Order Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE1)')
    plt.show()
    # plot hypothesis fit
    y = X_train@weights
    plt.plot(X_train[:,-1] , y, '-r', label=f'y={np.round(weights[0],2)}+{round(weights[1],2)}*x')
    plt.title('Final Model: First-order Univariate LR')
    plt.xlabel('GRE Score')
    plt.ylabel('Chance of Admission')
    plt.legend(loc='upper left')
    plt.scatter(X_train[:,-1], y_train)
    plt.show()      
        
    # measure performance on test set
    test_RMSE1 = np.sqrt((1/(2*m_test)) * np.transpose(((X_test@weights) - y_test))@\
                       ((X_test@weights) - y_test))
    print(f'Test RMSE1: {np.round(test_RMSE1,3)}\n')
    
# ==================== 2nd order polynomial regression ===================== #
if uni_second_order :
        
    # add feature: GRE Score^2 to the data for 2nd order polynomial
    X_train = np.insert(X_train,2,X_train[:,-1]**2, axis=1)
    X_test = np.insert(X_test,2,X_test[:,-1]**2,axis=1)
    # initialise weights to 0
    weights = np.zeros(X_train.shape[1])
    
    RMSE_values = []
    epoch_values = []
    for i in range(epochs):
        epoch_values += [i]
        # fit model and compute cost for ith epoch
        RMSE2 = np.sqrt((1/(2*m_train)) * np.transpose(((X_train@weights) - y_train))@\
                       ((X_train@weights) - y_train))
        # keep track of loss for ith epoch
        RMSE_values += [RMSE2]
        # update weights (works simultaneously)
        weights = weights - lr * ((1/m_train) * np.transpose(X_train)@((X_train@weights) - y_train)) # partial derivative of L2-norm wrt theta1
        if i % 1000 == 0 :
            print(f'Epoch: {i} Training RMSE2: {round(RMSE2,3)}')
    print(f'\nFinal Training RMSE2: {round(RMSE2,3)}\nFinal Parameters: {np.round(weights,3)}') 
    print(f"Extrinsic Parameters: Epochs: {epochs}, Learning Rate: {lr}")       
        
    # plotting loss curve
    plt.plot(epoch_values,RMSE_values,'-b',label='RMSE')
    plt.title('Uni 2nd Order: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE2)')
    plt.show()
            
    # measure performance on test set
    test_RMSE2 = np.sqrt((1/(2*m_test)) * np.transpose(((X_test@weights) - y_test))@\
                       ((X_test@weights) - y_test))
    print(f'Test RMSE2: {np.round(test_RMSE2,3)}\n')

# ================= Multivariate Linear Regression ========================= #

# ========================= For loop Implementation ======================== #
if multi_first_order and loop :
   
    # Train-Test Split
    # get all feature values and unused rows for vectorized implementation
    X_train = df3.iloc[:,1:-1][100:]
    y_train = df3['chance_of_admit'][100:]
    X_test = df3.iloc[:,1:-1][:100]
    y_test = df3['chance_of_admit'][:100]
    # initialise weights
    w_0 = 0
    w_1 = 0
    w_2 = 0
    w_3 = 0
    w_4 = 0
    w_5 = 0
    w_6 = 0
    w_7 = 0
    # training loop
    RMSE_values = []
    epoch_values = []
    start_time = time.time() # keep track of training time
    for i in range(epochs): 
        SE = []
        temp_values_b = []
        temp_values_w1 = []
        temp_values_w2 = []
        temp_values_w3 = []
        temp_values_w4 = []
        temp_values_w5 = []
        temp_values_w6 = []
        temp_values_w7 = []
        epoch_values += [i]
        for j in range(len(X_train)) :
            # fit model
            y_hat = w_0 + w_1 * X_train['gre_score'][100+j] + w_2 * X_train['TOEFL Score'][100+j] +\
                w_3 * X_train['University Rating'][100+j] + w_4 * X_train['SOP'][100+j] + \
                w_5 * X_train['LOR'][100+j] + w_6 * X_train['CGPA'][100+j] +\
                    w_7 * X_train['Research'][100+j]
            # compute squared loss
            SE += [(y_hat - y_train[100+j])**2]
            # keep track of values for bias update
            temp_values_b += [y_hat - y_train[100+j]]
            # keep track of values for weight1 update 
            temp_values_w1 += [(y_hat - y_train[100+j]) * X_train['gre_score'][100+j]]
            # keep track of values for weight2 update
            temp_values_w2 += [(y_hat - y_train[100+j]) * X_train['TOEFL Score'][100+j]]
            # keep track of values for weight3 update
            temp_values_w3 += [(y_hat - y_train[100+j]) * X_train['University Rating'][100+j]]
            # keep track of values for weight4 update
            temp_values_w4 += [(y_hat - y_train[100+j]) * X_train['SOP'][100+j]]
            # keep track of values for weight5 update
            temp_values_w5 += [(y_hat - y_train[100+j]) * X_train['LOR'][100+j]]
            # keep track of values for weight6 update
            temp_values_w6 += [(y_hat - y_train[100+j]) * X_train['CGPA'][100+j]]
            # keep track of values for weight7 update
            temp_values_w7 += [(y_hat - y_train[100+j]) * X_train['Research'][100+j]]
        # compute RMSE 
        RMSE3 = math.sqrt((1/(2*m_train)) * sum(SE)) # sqrt of L2-Norm Cost Function
        # keep track for plotting
        RMSE_values += [RMSE3]
        if i % 500 == 0 or i+1 == epochs :
            print(f"Epoch: {i} Training RMSE: {round(RMSE3,3)}")
        # preserve previous weights for reporting after last epoch
        final_bias = w_0
        final_weight1 = w_1
        final_weight2 = w_2
        final_weight3 = w_3
        final_weight4 = w_4
        final_weight5 = w_5
        final_weight6 = w_6
        final_weight7 = w_7
        # gradient descent optimization
        temp_w0 = (1/m_train) * sum(temp_values_b) # partial derivative of the loss function w.r.t. w0
        temp_w1 = (1/m_train) * sum(temp_values_w1) # partial derivative of the loss function w.r.t. w1
        temp_w2 = (1/m_train) * sum(temp_values_w2) # partial derivative of the loss function w.r.t. w2
        temp_w3 = (1/m_train) * sum(temp_values_w3) # partial derivative of the loss function w.r.t. w3
        temp_w4 = (1/m_train) * sum(temp_values_w4) # partial derivative of the loss function w.r.t. w4
        temp_w5 = (1/m_train) * sum(temp_values_w5) # partial derivative of the loss function w.r.t. w5
        temp_w6 = (1/m_train) * sum(temp_values_w6) # partial derivative of the loss function w.r.t. w6
        temp_w7 = (1/m_train) * sum(temp_values_w7) # partial derivative of the loss function w.r.t. w7
        # simultaneous update
        w_0 = w_0 - lr * temp_w0 # update bias
        w_1 = w_1 - lr * temp_w1 # update eight 1
        w_2 = w_2 - lr * temp_w2 # update weight 2
        w_3 = w_3 - lr * temp_w3 # update weight 3
        w_4 = w_4 - lr * temp_w4 # update weight 4
        w_5 = w_5 - lr * temp_w5 # update weight 5
        w_6 = w_6 - lr * temp_w6 # update weight 6
        w_7 = w_7 - lr * temp_w7 # update weight 6
    # return final parameters and training RMSE 
    end_time = time.time()
    tot_time = end_time - start_time
    print('\nTraining process is complete! Training time is: ', str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"+str(int((tot_time%3600)%60)))
    print(f"Final parameters: bias: {round(final_bias,4)} w1: {round(final_weight1,4)}, w2: {round(final_weight2,4)}")
    print(f"Final parameters: w3: {round(final_weight3,4)} w4: {round(final_weight4,4)}, w5: {round(final_weight5,4)}")
    print(f"Final parameters: w6: {round(final_weight6,4)} w7: {round(final_weight7,4)}")
    print(f"Extrinsic Parameters: Epochs: {epochs}, Learning Rate: {lr}")
    print(f'\nTraining RMSE: {round(RMSE3,3)}')
    
    # plotting loss curve
    plt.plot(epoch_values,RMSE_values,'-b',label='RMSE')
    plt.title('Multi 1st Order: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE3)')
    plt.show()
        
    # test performance on test set
    SE = []
    for i in range(len(X_test)) :
        # make predictions
        y_hat = w_0 + w_1 * X_test['gre_score'][i] + w_2 * X_test['TOEFL Score'][i] + \
                w_3 * X_test['University Rating'][i] + w_4 * X_test['SOP'][i] + \
                w_5 * X_test['LOR'][i] + w_6 * X_test['CGPA'][i] + w_7 * X_test['Research'][i]
        # compute squared loss
        SE += [(y_hat - y_test[i])**2]
    # compute RMSE 
    test_RMSE3 = math.sqrt((1/(2*m_test)) * sum(SE)) # sqrt of L2-Norm Cost Function
    print(f"Test RMSE: {round(test_RMSE3,4)}\n")
    
# ======================= Vectorized Implementation ======================== #
if multi_first_order and vectorized :
        
    # get all training feature values for vectorized implementation
    X_train = df3.iloc[:, :-1][100:].values
    y_train = df3['chance_of_admit'][100:].values
    # get test feature values for vectorized implementation
    X_test = df3.iloc[:, :-1][:100].values
    y_test = df3['chance_of_admit'][:100].values
    # initialise weights to 0
    weights = np.zeros(X_train.shape[1])
    
    RMSE_values = []
    epoch_values = []
    for i in range(epochs):
        start_time2 = time.time() # keep track of training time

        epoch_values += [i]
        # fit model and compute cost for ith epoch
        RMSE3 = np.sqrt((1/(2*m_train)) * np.transpose(((X_train@weights) - y_train))@\
                       ((X_train@weights) - y_train))
        # keep track of loss for ith epoch
        RMSE_values += [RMSE3]
        # update weights (works simultaneously)
        weights = weights - lr * ((1/m_train) * np.transpose(X_train)@((X_train@weights) - y_train)) # partial derivative of L2-norm wrt theta1
        if i % 1000 == 0 :
            print(f'Epoch: {i} Training RMSE3: {round(RMSE3,3)}')
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print('\nTraining process is complete! Training time is:', str(int((tot_time2/3600)))+":"+str(int((tot_time2%3600)/60))+":"+str(int((tot_time2%3600)%60)))
    print(f'Final Training RMSE3: {round(RMSE3,3)}\nFinal Parameters: {np.round(weights,3)}') 
    print(f"Extrinsic Parameters: Epochs: {epochs}, Learning Rate: {lr}")       
        
    # plotting loss curve
    plt.plot(epoch_values,RMSE_values,'-b',label='RMSE')
    plt.title('Multi 1st Order: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE3)')
    plt.show()
            
    # measure performance on test set
    test_RMSE3 = np.sqrt((1/(2*m_test)) * np.transpose(((X_test@weights) - y_test))@\
                       ((X_test@weights) - y_test))
    print(f'Test RMSE3: {np.round(test_RMSE3,3)}')

if loop and vectorized :
    x = ['For Loop', 'Vectorized']
    y = [tot_time,tot_time2]
    plt.bar(x,y)
    plt.title('Training Time Comparison',fontsize='x-large',weight='bold')
    plt.ylabel('Training Time (seconds)')
    plt.show()

# ================== Quick Model Comparison by RMSE ========================= #
if RMSE1 and RMSE2 and RMSE3 :
    models = ['uni_order1', 'uni_order2','multi_order1']
    performances = [RMSE1,RMSE2,RMSE3]
    plt.bar(models,performances)
    plt.title('Model Comparsion',fontsize='x-large',weight='bold')
    plt.xlabel('Models',fontsize='large',weight='bold')
    plt.ylabel('RMSE',fontsize='large',weight='bold')
    plt.show()
