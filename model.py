import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


#create a multilayer perceptron
def MLP_model(num_pixels, num_classes):
    model = Sequential()
    #the parts we will configure; number of dense layers and size of each one
    model.add(Dense(900, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#load training data
print("reading fashion-train.csv...")
df = pd.read_csv("fashion-train.csv")
X = df.drop(columns="label")
X = X.to_numpy()
X = np.reshape(X, (60000,784))
y = df["label"]
y = y.to_numpy()
y = np.reshape(y, (60000,1))

print("Splitting training data using k-fold cross validation...")
X, Xt, y, yt = train_test_split(X, y, test_size=0.3)

print("reading fashion-test.csv...")
dfTest = pd.read_csv("fashion-test.csv")
Xtest = dfTest
Xtest = Xtest.to_numpy()
Xtest = np.reshape(Xtest, (10000,784))
#X = X.to_numpy()
#X = np.reshape(Xt, (1836,1))

num_pixels = 784
print("normalizing data...")
#normalize data. 255 is maxpixel for greyscale 
X = X / 255.0 
Xt = Xt / 255.0

print("converting class labels to one-hot encoding...")
#convert class labels to one-hot encoding
y = np_utils.to_categorical(y)
#yt = np_utils.to_categorical(yt)

#get number of classes (10)
num_classes = y.shape[1]

print("creating multi-layer-perceptron model and fitting data...")
#create the MLP model and fit
mlp = MLP_model(num_pixels, num_classes)
mlp.fit(X, y, epochs=9)

#grab a random test sample (and answer)
i = np.random.randint(0, len(Xt)) #grab random index
xt = Xt[i,:] #grab random testing sample, give all columns of test sample i
ans = np.argmax(y[i,:]) #get position of correct answer for ith row of all the columns

#reshape the input to work with MLP
xt = xt.reshape(num_pixels,1).T

print("making prediction on split training data...")
#make prediction
yp = mlp.predict(xt)
#print answer
pred = np.argmax(yp)
print("answer =", ans, 'pred =', pred)


#visualize data
xt = xt.reshape(28,28)
import matplotlib.pyplot as plt
plt.imshow(xt)

print("making prediction on test data...")
yp = mlp.predict(Xtest)
p = []
for i in range(1,1000):
    yp = np.argmax(y[i,:])
    p.append(yp)
predictions = pd.DataFrame(p)

print("writing predictions to Swanson.csv")
pd.DataFrame(predictions).to_csv('Swanson.csv', header=None, index=None)

