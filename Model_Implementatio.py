import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import random
from tkinter import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from numpy import array
from sklearn.preprocessing import MinMaxScaler

global bias_suma, bias, cost, activ_hype, acum, h, error, epochs, count, c_test, values, activ_hyp, active_hyp_train, activ_hyp_test, content, df_y, df_temp
params = np.array([0, 0, 0.1, 0, 0, 0])
values = []
activ_hyp = []
activ_hyp_train = []
activ_hyp_test = []
content = []
c_test = 0
bias_suma = 0
suma = 0
bias = 0
cost = 0
acum = 0
h = 0
b = 0
alpha = 0.01



columns = ['Temp','NOcc','Lpain','Urinneed','Mpain','Udisc','UBinf','Nephritis']
data = pd.read_csv("/Users/Christian Sámano/Desktop/diagnosis.csv", names = columns)

data = data.drop([0], axis=0)

dff = pd.DataFrame(data)

scaler = MinMaxScaler()
df = scaler.fit_transform(dff.to_numpy())
df = pd.DataFrame(df, columns=[
  'Temp','NOcc','Lpain','Urinneed','Mpain','Udisc','UBinf','Nephritis'])
 
shuffle = df.sample(frac = 1)
shuffle_x = shuffle[["Temp", "NOcc", "Lpain", "Urinneed", "Mpain", "Udisc"]]
shuffle_y = shuffle[["Nephritis"]]

shuffle_x = np.array(shuffle_x)
shuffle_y = np.array(shuffle_y)
shuffle_temp = df['Temp']

df_x = df[["Temp", "NOcc", "Lpain", "Urinneed", "Mpain", "Udisc"]]
df_y = df[["Nephritis"]]

df_x = np.array(df_x)
df_y = np.array(df_y)

df_x_train = df_x[:-20]
df_y_train = df_y[:-20]
df_x_test = df_x[-20:]
df_y_test = df_y[-20:]


df_temp = df['Temp']
df_temp_train = df_temp[:-20]
df_temp_test = df_temp[-20:]

window = Tk()
window.geometry("500x300+100+100")
window.title("Nephritis of Renal Pelvis Predictor")

"""Templbl = Label(window, text= "Temp").place(x = 20, y = 220)
txtTemp = Entry(window, textvariable = Tempval, width = 5).place(x = 20, y = 250)

NOcclbl = Label(window, text= "NOcc").place(x = 80, y = 220)
txtNOcc = Entry(window, textvariable = NOccval, width = 5).place(x = 80, y = 250)

Lpainlbl = Label(window, text= "Lpain").place(x = 140, y = 220)
txtLpain = Entry(window, textvariable = Lpainval, width = 5).place(x = 140, y = 250)

Urinneedlbl = Label(window, text= "Urinneed").place(x = 200, y = 220)
txtUrinneed = Entry(window, textvariable = Urinneedval, width = 5).place(x = 200, y = 250)

Mpainlbl = Label(window, text= "Mpain").place(x = 260, y = 220)
txtMpain = Entry(window, textvariable = Mpainval, width = 5).place(x = 260, y = 250)

Udisclbl = Label(window, text= "Udisc").place(x = 320, y = 220)
txtUdisc = Entry(window, textvariable = Udiscval, width = 5).place(x = 320, y = 250)"""

for i in range(len(df_y)):
    df_y[i] = float(df_y[i])

def hyp(params, samples, b):
    global acum, h, x, count_pred
    acum = []
    count_pred = []
    x=0
    for i in range(len(samples)):
        h = 0
        count_pred = np.append(count_pred, x)
        x = x + 1
        for j in range (len(params)):
            h = h + (params[j] * float(samples[i][j]))
        h = h + b
        acum = np.append(acum, h)
        
def hyp2(params, samples, b):
    global h
    h = 0
    print(params)
    for i in range(len(params)):   
        h = h + (params[i] * float(samples[i]))
        print(str(params[i]) + '...' + str(samples[i]))
    h = h + b
    print(h)
    sigmoidh = 1 / (1 + math.exp(-h))
    if sigmoidh > 0.5:
        sigmoidh = 1.0
    else:
        sigmoidh = 0.0
    return sigmoidh

def activ(hypothesis):
    global activ_hyp
    sigmoid = 0
    activ_hyp = []
    for i in range(len(hypothesis)):
        sigmoid = 1 / (1 + math.exp(-hypothesis[i]))
        activ_hyp = np.append(activ_hyp, sigmoid)
        
def cross_en(instances, hypothesis, realval):
    global uni_cost, cost, unit_error
    cost = 0
    unit_cost = 0
    unit_error = []
    for i in range(instances):
        if  (realval[i]) == 1:
            unit_cost = (-float(realval[i]) * math.log(float(hypothesis[i])))
            cost = cost + (-float(realval[i]) * math.log(float(hypothesis[i])))
        if  (realval[i]) == 0:
            unit_cost = (-(1 - float(realval[i])) * math.log(1 - float(hypothesis[i])))
            cost = cost + (-(1 - float(realval[i])) * math.log(1 - float(hypothesis[i])))    
        unit_error.append(unit_cost)
    cost = cost / instances
def params_gd(params, alpha, instances, hypothesis, realval, features):
    global suma
    for i in range(len(params)):
        suma = 0
        for j in range(instances):
            suma = suma + (((float(hypothesis[j]) - float(realval[j]))) * float(features[j][i]))
        suma = (suma * alpha) / instances
        params[i] = params[i] - suma
        
def bias_gd(b, alpha, instances, hypothesis, realval):
    global bias_suma, bias
    bias_suma = 0
    bias = b
    for j in range(instances):
        bias_suma = bias_suma + (((float(hypothesis[j]) - float(realval[j]))) * 1)
    bias_suma = (bias_suma * alpha) / instances
    bias = bias - bias_suma
    return b
    
def roundup(values):
    for i in range(len(values)):
        if values[i] > 0.5:
            values[i] = 1.0
        else:
            values[i] = 0.0
            
def train():
    global epochs, count_train, error_train, activ_hyp_train, flag
    flag = 1
    epochs = 0
    count_train = []
    error_train = []
    activ_hyp_train = []
    while (epochs != 10000 and flag != 0): 
        instances = len(df_x_train)
        hyp(params, df_x_train, bias)
        activ(acum)
        activ_hyp_train = activ_hyp
        cross_en(instances, activ_hyp_train, df_y_train)
        params_gd(params, alpha, instances, activ_hyp_train, df_y_train, df_x_train)
        "print(params)"
        bias_gd(bias, alpha, instances, activ_hyp_train, df_y_train)
        count_train = np.append(count_train, epochs)
        epochs = epochs + 1
        "print(cost)"
        if cost > 0.3:
            flag = 1
        else:
            flag = 0
        error_train = np.append(error_train, cost)
    roundup(activ_hyp_train)
    Traindone = Label(window, text= "Train Done! ").place(x = 400, y = 20)
def test():
    global epochs, count_test, error_test, activ_hyp_test, flag
    flag = 1
    epochs = 0
    count_test = []
    error_test = []
    activ_hyp_test = []
    while (epochs != 10000 and flag != 0 ): 
        instances = len(df_x_test)
        hyp(params, df_x_test, bias)
        activ(acum)
        activ_hyp_test = activ_hyp
        cross_en(instances, activ_hyp_test, df_y_test)
        params_gd(params, alpha, instances, activ_hyp_test, df_y_test, df_x_test)
        "print(params)"
        bias_gd(bias, alpha, instances, activ_hyp_test, df_y_test)
        count_test = np.append(count_test, epochs)
        epochs = epochs + 1
        "print(cost)"
        if cost > 0.25:
            flag = 1
        else:
            flag = 0
        error_test = np.append(error_test, cost)
    roundup(activ_hyp_test)
    Testdone = Label(window, text= "Test Done! ").place(x = 400, y = 60)
    
def cfmatrix():
    global df_y_train_, activ_hyp_train_
    df_y_train_ = []
    activ_hyp_train_ = []
    for i in range(len(shuffle_y)):
        df_y_train_ = np.append(df_y_train_, int(shuffle_y[i]))
        activ_hyp_train_ = np.append(activ_hyp_train_, int(activ_hyp_pred2[i]))
        
    y_true = df_y_train_
    y_pred = activ_hyp_train_
    
    cf_matrix = confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Nephritis of Renal Pelvis Predictor Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    plt.show()

def plots():
    """roundup(activ_hyp_train)
    roundup(activ_hyp_test)
    roundup(activ_hyp_pred)"""
    'fig, axis = plt.subplots(3, 2)'

    plt.title("Test Error")
    plt.plot(count_test, error_test)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.show()
    plt.title("Train Error")
    plt.plot(count_train, error_train)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    '''axis[0, 0].set_title("error_train")
    axis[0, 1].plot(count_test, error_test)
    axis[0, 1].set_title("error_test")
    axis[1, 0].scatter(df_temp_train, activ_hyp_train)
    axis[1, 0].set_title("temp-train")
    axis[1, 1].scatter(df_temp_test, activ_hyp_test)
    axis[1, 1].set_title("temp-test")
    axis[2, 0].scatter(df_temp_train, df_y_train)
    axis[2, 0].set_title("temp-real")
    axis[2, 1].scatter(df_temp_test, df_y_test)
    axis[2, 1].set_title("temp-real")'''

    """axis[2, 0].scatter(df_temp, activ_hyp_pred)
    axis[2, 0].set_title("temp-predict")
    axis[2, 1].scatter(count_pred, error_pred)
    axis[2, 1].set_title("error_prediction")"""
    
    plt.show()

def predict():
    
    global activ_hyp_pred, activ_hyp_pred2, predictors, error_pred, instances, count_pred
    predictors2 = [0,0,0,0,0,0]
    predictors = []
    activ_hyp_pred = []
    activ_hyp_pred2 = []
    error_pred = 0
    instances = len(shuffle_y)
    predictors = shuffle_x
    hyp(params, predictors, bias)
    activ(acum)
    activ_hyp_pred2 = activ_hyp
    cross_en(instances, activ_hyp_pred2, shuffle_y)
    error_pred = unit_error

    print(bias)
    roundup(activ_hyp_pred2)

    
    predictors2[0] = txtTemp.get()
    predictors2[1] = txtNOcc.get()
    predictors2[2] = txtLpain.get()
    predictors2[3] = txtUrinneed.get()
    predictors2[4] = txtMpain.get()
    predictors2[5] = txtUdisc.get()

    predictors2[0] = (float(predictors2[0]) - 35.5) / (41.5 - 35.5)

    error_pred = 0
    print(predictors2)
    activ_hyp_pred = hyp2(params, predictors2, bias)
        
    
    Predres = Label(window, text= "Result: " + str(activ_hyp_pred)).place(x = 400, y = 180)

    '''fpr, tpr, _ = roc_curve(df_y_train, activ_hyp_train)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="Train ROC curve (area = %i)" %roc_auc,
    )
    plt.show()'''
    '''fpr, tpr, _ = metrics.roc_curve(df_y_test,  activ_hyp_test)
    auc = metrics.roc_auc_score(df_y_test, activ_hyp_test)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.title('Test ROC curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()'''
    '''PrecisionRecallDisplay.from_predictions(shuffle_y, activ_hyp_pred2)
    plt.show()
    rscore = recall_score(shuffle_y, activ_hyp_pred2, average=None)
    print(rscore)'''
    
trainbtn = Button(window, text = "Train", command = train, width = 10).place(x = 20, y = 20)
testbtn = Button(window, text = "Test", command = test, width = 10).place(x = 20, y = 60)
cfmatrixbtn = Button(window, text = "Cf Matrix", command = cfmatrix, width = 10).place(x = 20, y = 100)
Plotbtn = Button(window, text = "Plot", command = plots, width = 10).place(x = 20, y = 140)
Predbtn = Button(window, text = "Predict", command = predict, width = 10).place(x = 20, y = 180)

Templbl = Label(window, text= "Temp").place(x = 20, y = 220)
txtTemp = Entry(window, width = 5)
txtTemp.place(x = 20, y = 250)

NOcclbl = Label(window, text= "NOcc").place(x = 80, y = 220)
txtNOcc = Entry(window, width = 5)
txtNOcc.place(x = 80, y = 250)

Lpainlbl = Label(window, text= "Lpain").place(x = 140, y = 220)
txtLpain = Entry(window, width = 5)
txtLpain.place(x = 140, y = 250)

Urinneedlbl = Label(window, text= "Urinneed").place(x = 200, y = 220)
txtUrinneed = Entry(window, width = 5)
txtUrinneed.place(x = 200, y = 250)

Mpainlbl = Label(window, text= "Mpain").place(x = 260, y = 220)
txtMpain = Entry(window, width = 5)
txtMpain.place(x = 260, y = 250)

Udisclbl = Label(window, text= "Udisc").place(x = 320, y = 220)
txtUdisc = Entry(window, width = 5)
txtUdisc.place(x = 320, y = 250)

window.mainloop()



        
