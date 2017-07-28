import pandas as pd
import math
import numpy as np

#get the dataset ready
dataset=pd.read_csv("data2.csv")
X=dataset.iloc[:,:-1].values
X=np.append(np.ones((X.shape[0],1)),X,1)
y=dataset.iloc[:,-1].values

#splitting to train/test sets:
test_to_train_ratio=0.2 #20% of training set
test_size=int(X.shape[0]*test_to_train_ratio)
train_size=int(X.shape[0]-test_size)
X_train=X[0:train_size,:]
y_train=y[0:train_size]
m=X_train.shape[0]
n=X_train.shape[1]
X_test=X[train_size:test_size+train_size,:]
y_test=y[train_size:test_size+train_size]

"""Note that the train and the test set will not be used here. whole of X will be uesd to train"""

def normalize(X):
    for feature in range(1,X.shape[1]):
        sum=np.sum(X[:,feature])
        mean=sum/X.shape[0]
        X[:,feature]-=mean
        X[:,feature]/=mean
    return X

X=normalize(X)

def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except OverflowError as e:
        return 0.000000000000001
#vectorize sigmoid function
#sigmoid=np.vectorize(sigmoid)

#initialize weights
theta=np.zeros(X.shape[1]) #the weights, I call em theta with a bias included

#prediction function returns a list/1D array containing zeros and ones
def predict(x,theta):
    z=np.matmul(x,theta)
    return (z+abs(z))/(z*2)

def predict2(x,theta):
    z=np.matmul(x,theta)
    return sigmoid(z)

def cost_function(x,theta,y):         
    #returns cost for one training_example
    #x and theta are vectors and y is number 1 or 0
    return y*(math.log((1-sigmoid(np.matmul(x,theta)))/sigmoid(np.matmul(x,theta)))) - math.log(1-sigmoid(np.matmul(x,theta)))

def cost_gradient(x,theta,y,j):
    #print("at ",j," :  ",sigmoid(np.matmul(x,theta)))
    return (sigmoid(np.matmul(x,theta))-y)*x[j]
    
def gradient_descent(X,y,theta,learning_rate=10,batch_size=1,num_epochs=100):
    for epoch in range(num_epochs):
        
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        for i in range(int(X.shape[0]/batch_size)):
            cost=0
            grads=[0.0,0.0,0.0]
            for j in range(n):
                descent=0
                for k in range(batch_size):
                    descent+=cost_gradient(X[(batch_size*i)+k],theta,y[(batch_size*i)+k],j)
                grads[j]=descent/batch_size
            print(grads)
            theta[0]=theta[0]-learning_rate*(grads[0])
            theta[1]=theta[1]-learning_rate*(grads[1])
            theta[2]=theta[2]-learning_rate*(grads[2])
            print(theta)
            for _ in range(batch_size):
                cost+=cost_function(X[(batch_size*i)+_],theta,y[(batch_size*i)+_])
            cost=cost/batch_size
            print("cost:  {}".format(cost))


#now train it using a particular batch size and learning rate
gradient_descent(X,y,theta,learning_rate=10,batch_size=100,num_epochs=800)
    
"""finally the cost comes out to be 0.2034 approx, which is correct"""

    
"""'expected theta: [-25.161, 0.206, 0.201], without normalization"""
"""check answer on x= [1, 45, 85]"""

predict2(np.array([1,(45-65.644274057323145)/65.644274057323145,(85-66.221998088116948)/66.221998088116948]),theta)
    
"""answer is 0.775, which is correct"""


#finally to find accuracy on training set:

y_pred=np.zeros((X.shape[0],))

for _ in range(X.shape[0]):
    y_pred[_]=predict(X[_],theta)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y,y_pred)

accuracy=(cm[0,0]+cm[1,1])/X.shape[0]

print("Accuracy = {}% ,on the training set.".format(accuracy*100))


"""
On training with mini batch grad_desc with batch size of 10, the accuracy was found 
to be 92.0% on X. While on batch_size 100, the accuracy was 89.0%.

"""






