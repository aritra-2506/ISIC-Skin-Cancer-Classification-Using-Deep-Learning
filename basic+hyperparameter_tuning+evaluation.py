import cv2
import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers
from keras import models
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import GridSearchCV

#Load image files from directory
filenames = glob.glob("C:/Users/Aritra Mazumdar/Downloads/ISIC/img/*.jpg")
filenames.sort()
images_list = [cv2.imread(img) for img in filenames]

#Data Preprocessing - Resize & Rescale images
X=[]
for img in images_list:
    X.append(cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC))
images=np.asarray(X)
images = images.astype('float32') / 255

#Load labels
labels_df = pd.read_csv('C:/Users/Aritra Mazumdar/Downloads/ISIC/ISIC_2019_Training_GroundTruth.csv')

#Data Preprocessing - labels
labels=labels_df.to_numpy()
labels=labels[:,1:]
labels=labels[0:10,:]
labels=labels.astype('float32')

#Train-Test split
images_train=images[0:8]
images_test=images[8:10]

labels_train=labels[0:8]
labels_test=labels[8:10]

#Building & Compiling model for Grid Search
def create_model(optimizer='adam', activation='relu', dropout_rate=0.0):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(299, 299, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation=activation))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(9, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [100, 500, 1000]
epochs = [5, 10, 20, 50]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(images_train, labels_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
x=grid_result.best_params_
a,b,c,d,e=x.values()

#Building model with best parameters
model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(299, 299, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(9, activation=a))
model.add(layers.Dropout(rate=c))
model.add(layers.Dense(9, activation='softmax'))

#Compiling
model.compile(loss='categorical_crossentropy', optimizer=e, metrics=['accuracy'])

#Fitting
hist = model.fit(images_train, labels_train,
           batch_size=b, epochs=d, validation_split=0.3 )

#Loss & Accuracy
test_loss, test_acc = model.evaluate(images_test, labels_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#Plot Accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.clf()

#Plot Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Test an image
probabilities = model.predict(np.array( [images_test[1],] ))
print(probabilities)

plt.imshow(images_test[1], cmap=plt.cm.binary)

number_to_class = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion', 'Squamous Cell Carcinoma', 'None of the others']

index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Second most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Third most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fourth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])
print("Fifth most likely class:", number_to_class[index[4]], "-- Probability:", probabilities[0,index[4]])

#Define Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()

# Predict the values from the test dataset
Y_pred = model.predict(images_test)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(labels_test, axis = 1)
# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# Plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

#Save and Reload model
model.save('my_model.h5')
plot_model(model, to_file='my_model.png')
model = load_model('my_model.h5')

