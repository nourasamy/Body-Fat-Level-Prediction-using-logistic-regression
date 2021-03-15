import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("C:\\Users\\Mazen\\Downloads\\Bio-ML-Assignment-1\\Bodyfat-Levels.csv")
actual=data['bodyfatlevel']
data = data.drop(['bodyfatlevel', 'level'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,actual , test_size=0.1, random_state=4)

qq=x_test.to_csv('C:\\Users\\Mazen\\.spyder-py3\\Bodyfat-Level-TestData.csv')
#C:\\Users\\Mazen\\.spyder-py3\\BodyfatPercentage-Test-Data.csv
data2 = pd.read_csv("C:\\Users\\Mazen\\.spyder-py3\\Bodyfat-Level-TestData.csv")
data2 = data2.drop(['Unnamed: 0'], axis=1)
X = np.matrix(x_train.values)
X_testt=np.matrix(data2.values)

def Y_handling(YY):
    actual = YY
    Y_Essentialfat = []
    Y_Athletes = []
    Y_Fitness = []
    Y_Acceptable = []
    Y_Obesity = []
    for i in actual:
        if (i == 'Essential fat'):
            Y_Essentialfat.append(1)
            Y_Athletes.append(0)
            Y_Fitness.append(0)
            Y_Acceptable.append(0)
            Y_Obesity.append(0)
        elif (i == 'Athletes'):
            Y_Athletes.append(1)
            Y_Essentialfat.append(0)
            Y_Fitness.append(0)
            Y_Acceptable.append(0)
            Y_Obesity.append(0)
        elif (i == 'Fitness'):
            Y_Fitness.append(1)
            Y_Essentialfat.append(0)
            Y_Athletes.append(0)
            Y_Acceptable.append(0)
            Y_Obesity.append(0)
        elif (i == 'Acceptable'):
            Y_Acceptable.append(1)
            Y_Essentialfat.append(0)
            Y_Fitness.append(0)
            Y_Athletes.append(0)
            Y_Obesity.append(0)
        else:
            Y_Obesity.append(1)
            Y_Essentialfat.append(0)
            Y_Fitness.append(0)
            Y_Athletes.append(0)
            Y_Acceptable.append(0)
    Y_Essentialfat = np.matrix(Y_Essentialfat)
    Y_Athletes = np.matrix(Y_Athletes)
    Y_Fitness = np.matrix(Y_Fitness)
    Y_Acceptable = np.matrix(Y_Acceptable)
    Y_Obesity = np.matrix(Y_Obesity)
    return Y_Essentialfat ,Y_Athletes ,Y_Fitness ,Y_Acceptable,Y_Obesity


def normalize(X):

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X) / rng)
    return norm_X


def hyposis_func(beta, X):

    return 1.0 / (1 + np.exp(-np.dot(X, beta.T)))


def log_gradient(beta, X, y):

    first_calc = hyposis_func(beta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    return final_calc


def cost_func(beta, X, y):

    log_func_v = hyposis_func(beta, X)
    y = np.squeeze(y)
    first = y * np.log(log_func_v)
    second = (1 - y) * np.log(1 - log_func_v)
    final = -first - second
    return np.mean(final)


def gradient_func(X, y, beta, lr=.00001, converge_change=.0001):

    cost = cost_func(beta, X, y)
    change_cost = 1
    num_iter = 1

    while (change_cost > converge_change):
        old_cost = cost
        beta = beta - (lr * log_gradient(beta, X, y))
        cost = cost_func(beta, X, y)
        change_cost = old_cost - cost
        num_iter += 1

    return beta


X = normalize(X)
X_testt=normalize(X_testt)
X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
X_testt = np.hstack((np.matrix(np.ones(X_testt.shape[0])).T, X_testt))
beta = np.matrix(np.zeros(X.shape[1]))
Y_Essentialfat ,Y_Athletes ,Y_Fitness ,Y_Acceptable,Y_Obesity=Y_handling(y_train)

#beta_Essentialfat = gradient_func(X, Y_Essentialfat, beta,0.03)
#beta_Athletes = gradient_func(X, Y_Athletes,beta,.009,.00001)
#beta_Fitness = gradient_func(X, Y_Fitness, beta,0.01)
#beta_Acceptable = gradient_func(X, Y_Acceptable, beta,0.0001,0.00001)
#beta_Obesity = gradient_func(X, Y_Obesity, beta,0.01)

beta_Essentialfat = gradient_func(X, Y_Essentialfat, beta)
beta_Athletes = gradient_func(X, Y_Athletes,beta)
beta_Fitness = gradient_func(X, Y_Fitness, beta)
beta_Acceptable = gradient_func(X, Y_Acceptable, beta)
beta_Obesity = gradient_func(X, Y_Obesity, beta)



def pred_values(X_test):

    pred_prob_Acceptable = hyposis_func(beta_Acceptable, X_test)
    pred_prob_Athletes = hyposis_func(beta_Athletes, X_test)
    pred_prob_Essentialfat = hyposis_func(beta_Essentialfat, X_test)
    pred_prob_Fitness = hyposis_func(beta_Fitness, X_test)
    pred_prob_Obesity = hyposis_func(beta_Obesity, X_test)
    li=[]
    for i in range(len(pred_prob_Acceptable)):
        maxs=max(pred_prob_Acceptable[i],pred_prob_Athletes[i],pred_prob_Essentialfat[i],pred_prob_Fitness[i],pred_prob_Obesity[i])
        if(maxs==pred_prob_Acceptable[i]):
            li.append('Acceptable')
        elif(maxs==pred_prob_Athletes[i]):
            li.append('Athletes')
        elif(maxs==pred_prob_Essentialfat[i]):
            li.append('Essentialfat')
        elif(maxs==pred_prob_Fitness[i]):
            li.append('Fitness')
        elif(maxs==pred_prob_Obesity[i]):
            li.append('Obesity')
    return li

y_pred = pred_values(X_testt)
ss=np.sum(np.matrix(y_pred)==np.matrix(y_test))
print("Accurecy {} %".format((ss/float(len(y_pred)))*100))
print("=================================================")
print(list(y_test))
print("=================================================")
print(y_pred)