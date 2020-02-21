import cv2
import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import plot_model

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
labels=labels[0:6007,:]
labels=labels.astype('float32')

#Train-Test split
images_train=images[0:5000]
images_test=images[5000:6007]

labels_train=labels[0:5000]
labels_test=labels[5000:6007]

#Building model
model = Sequential()
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(299,299,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(150))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=.3))

model.add(Dense(250))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=.4))

model.add(Dense(9))
model.add(BatchNormalization())
model.add(Activation('softmax'))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#Compiling
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', precision_m, recall_m, f1_m])

#Fitting
hist = model.fit(images_train, labels_train,
           batch_size=50, epochs=25, validation_split=0.18)

#Test prediction
y_pred=model.predict(images_test)

#Test metrics evaluations
test_loss, test_accuracy, test_f1_score, test_precision, test_recall = model.evaluate(images_test, labels_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('Test f1 score:', test_f1_score)
print('Test precision:', test_precision)
print('Test recall:', test_recall)

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

plot_model(model, to_file='my_model.png')
