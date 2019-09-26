import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn

# read in the data
lines = []
with open('training_data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for l in reader:
        lines.append(l)

# split the data set        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)        

# generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                if source_path == 'center':
                    continue
                else:
                    file_name = source_path.split('G\\')[-1]
                    current_path = 'training_data/data/IMG/'+file_name
                    center_image = plt.imread(current_path)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle * -1)

                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)    
    
#     images = []
#     measurements = []
#     for l in lines:
#         source_path = l[0]
#         if source_path == 'center':
#             continue
#         else:
#             file_name = source_path.split('/')[-1]
#             current_path = 'sample_data/IMG/'+file_name
#             print(current_path)
#             image = plt.imread(current_path)
#             images.append(image)
#             measurement = float(l[3])
#             measurements.append(measurement)


#     X_train = np.array(images)
#     y_train = np.array(measurements)                                  

# hyperperemeter
batch_size=32

# compile and train the model using with generator
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# model architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))                                  
model.add(Conv2D(24,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(36,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(48,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Conv2D(64,(3,3))) 
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1))                                  

# compile and visualize the process
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# save the model
model.save('model.h5')
print('model saved')