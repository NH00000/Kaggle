import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns
# %matplotlib inline

#np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


#---------------Data preparation---------------#

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 


# Y_train.value_counts()
# X_train.isnull().any().describe()
# test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
random_seed=2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)

#---------------CNN---------------#

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(lr=0.0001)

model.compile(optimizer = adam , loss = "categorical_crossentropy", metrics=["accuracy"])

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=3,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.00001)

# epochs = 50 # Turn epochs to 30 to get 0.9967 accuracy
# batch_size = 86

model.fit(X_train, Y_train, epochs=5, batch_size=100)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_val, Y_val)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("data/cnn_mnist_datagen.csv",index=False)