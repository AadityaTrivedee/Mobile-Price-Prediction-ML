import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Extracting data
data = pd.read_csv('Cellphone.csv')
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

#Normalising Data to have the domain between [0,1]
def normalize(var1,var2):
    var1_temp,var2_temp = np.array(var1),np.array(var2)
    var2_temp_max = np.max(var2_temp,0)
    var2_temp_min = np.min(var2_temp,0)
    var2_temp_range = var2_temp_max - var2_temp_min
    return np.divide((var1_temp - var2_temp_min), var2_temp_range) 

# function to denormalise to unpack prediction value
def  denormalize(var1,var2):
    var2_temp = np.array(var2)
    var2_temp_max = np.max(var2_temp,0)
    var2_temp_min = np.min(var2_temp,0)
    var2_temp_range = var2_temp_max - var2_temp_min
    return float(np.multiply(var1, var2_temp_range) + var2_temp_min)

#Initialising parameters
X_train, X_test, y_train, y_test = train_test_split(normalize(X,X), normalize(y,y), test_size=0.4, random_state=1)
X_train, X_test, y_train, y_test = np.array(X_train, dtype = np.float64), np.array(X_test, dtype = np.float64), np.array(y_train, dtype = np.float64).reshape(-1,1), np.array(y_test, dtype = np.float64).reshape(-1,1)
X_train, X_test, y_train, y_test = X_train.astype("float64"), X_test.astype("float64") , y_train.astype("float64") , y_test.astype("float64")
dim = int(X_train.shape[1])
w,b,J, J_plt  = np.random.rand(dim,1)*0.001 , float(0) , float(0), []
w = np.array(w, dtype = np.float64)


print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)

print("Y_train Shape:", y_train.shape)
print("Y_test Shape: ",y_test.shape)
print("w Shape: ",w.shape)


def scatter_plt():
    for i in  range(1,dim+1):
        plt.scatter(X_train[...,i-1],y_train)
    plt.xlabel('Features_normalised')
    plt.ylabel('Price_normalised')
    plt.legend(data.columns[:-1])
    plt.grid()

def hypo(x):
    return np.dot(x, w)+ b

def algo(alpha):
    global w,b,J
    h = hypo(X_train)
    J = (1/(2*dim))*np.sum((h-y_train)**2)
    J_plt.append(J)
    dw, db = alpha*(1/dim)*(np.dot(X_train.T,(h-y_train))), alpha*(1/dim)*(np.sum(h-y_train))
    w -= dw
    b -= db

def cost_plt(itera):
    global J_plt
    x = np.array(np.linspace(1,itera,itera)).reshape(-1,1)
    J_plt = np.array(J_plt).reshape(-1,1)
    plt.plot(x,J_plt)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost function')
    plt.grid()
    plt.show()

def line_eqn():
    scatter_plt()
    llx = np.tile(np.array([np.array(np.arange(0,1,0.01))]).T, (1, dim))
    H = np.dot(llx,w)+b
    plt.plot(llx,H,"g")
    plt.grid()
    plt.show()

def accuracy():
    y_pred = hypo(X_test)
    return 1 - ((1/int(y_test.shape[0]))*(np.sum((y_pred - y_test)**2)))**(1/2)

def predict():
    features = data.columns[:-1].tolist()
    given = []
    for i in range(1,dim+1):
        a =  float(input(f"Enter the required value for {features[i-1]}: "))
        given.append(a)
    x_pred = normalize(given,X)
    result =  hypo(x_pred)
    print(f"The predicted output is: RS {round(denormalize(result,y) * 10,3)}\n")
    

def func(alpha, itera):
    scatter_plt()
    plt.show()
    for i in range(1,itera+1):
        algo(alpha)
        if i%(itera/10) == 0:    
            print(f"J = {J}\n")
            print(f"w = {w}\n")
            print(f"b = {b}\n")
    cost_plt(itera)
    line_eqn()
    print(f"The final Cost: {J}\n")
    print(f"The final W: {w}\n")
    print(f"The final b: {b}\n")
    print(f"Accuracy of the model is {round(accuracy() * 100,3)}%")
    predict()

func(0.06,200)