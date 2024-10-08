import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D,Dense,Flatten,AveragePooling2D,Dropout,BatchNormalization
from keras.models import Sequential , Model
from keras.applications import ResNet50,VGG16


class DL_model:
    def __init__(self,Age_gender_folder,emotion_folder):
        self.Age_gender_folder = Age_gender_folder
        self.emotion_folder = emotion_folder

    def  load_data(self,gender_batch:int,emotion_batch:int):
        D1 = self.Age_gender_folder

        age = []
        img = []
        gender = []

        for filename in os.listdir(D1):
            words = filename.split('_')
            age.append(int(words[0]))
            gender.append(int(words[1]))
            img.append(filename)

        df = pd.DataFrame({'age':age,'gender':gender,'img':img})

        dataset_age_gender = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            height_shift_range=0.2,
            width_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.1,
            rotation_range=10,
        )

        dataset_age_gender_generator = dataset_age_gender.flow_from_dataframe(df,directory=D1,x_col='img',y_col=['age','gender'],target_size=(150,150),batch_size=gender_batch,class_mode='multi_output')

        D2 = self.emotion_folder

        dataset_emotion = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.3,
            height_shift_range=0.2,
            width_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
        )

        dataset_emotion_generator = dataset_emotion.flow_from_directory(D2, target_size=(100,100),batch_size=emotion_batch,class_mode='categorical')
        
        return dataset_age_gender_generator,dataset_emotion_generator
    

    def model_age_gender(self,modal = None):
        """Train the deep learning model using the data generated by DataGeneration class"""

        # Load train and test datasets from    
        if(modal=='ResNet'):
            print("Resnet training")
            res = ResNet50(include_top=False,weights='imagenet',input_shape=(150,150,3))
            res.trainable = False

            output = res.layers[-1].output
        else:
            input = Input(shape=(150,150,3))
            conv1 = Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu')(input)
            max1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv1)

            conv2 = Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu')(max1)
            max2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv2)

            conv3 = Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu')(max2)
            output = MaxPooling2D(pool_size=(2,2),strides=2)(conv3)


        x= Flatten()(output)
        x1 = Dense(units=256,activation='relu')(x)
        w = BatchNormalization()(x1)
        z = Dense(units=256,activation='relu')(w)
        y1 = BatchNormalization()(z)

        x2 = Dense(units=256,activation='relu')(x)
        w1 = BatchNormalization()(x2)
        z1 = Dense(units=256,activation='relu')(w1)
        y2 = BatchNormalization()(z1)

        output1 = Dense(units=1,activation='linear',name='age')(y1)
        output2 = Dense(units=1,activation='sigmoid',name='gender')(y2)

        if(modal=='ResNet'):
            model = Model(inputs=res.input,outputs=[output1,output2])
        else:
            model = Model(inputs=input,outputs=[output1,output2])

        return model

    def model_emotion(self,modal=None):
        if modal=="ResNet":
            print("Resnet training")
            res = ResNet50(include_top=False,weights='imagenet',input_shape=(100,100,3))
            res.trainable = False
            
            model = Sequential()
            model.add(res)
            model.add(Flatten())
            model.add(Dense(128,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(7,activation='softmax'))

        else:

            model = Sequential()

            model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='tanh',input_shape=(100,100,3)))
            model.add(BatchNormalization())
            # model.add(MaxPooling2D(pool_size=(2,2),strides=1))

            model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='tanh'))
            model.add(BatchNormalization())
            # model.add(MaxPooling2D(pool_size=(2,2),strides=1))

            model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='tanh'))
            model.add(BatchNormalization())
            # model.add(MaxPooling2D(pool_size=(2,2),strides=1))

            model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='tanh'))
            model.add(BatchNormalization())
            # model.add(MaxPooling2D(pool_size=(2,2),strides=1))


            model.add(Flatten())
            model.add(Dense(64,activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(64,activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(64,activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(7,activation='softmax'))

        return model
    
    def compile_Fit_And_save(self,modal=None,gender_batch=32,emotion_batch=2):

        D1,D2 = self.load_data(gender_batch=gender_batch,emotion_batch=emotion_batch)

    

        # print("==================AGE Gender MODEL=====================")
        # M1 = self.model_age_gender(modal=modal)
        # M1.compile(optimizer='adam',loss={'age':'mae','gender':'binary_crossentropy'},metrics={'age':'mae','gender':'accuracy'})
        # M1.fit(D1,epochs=100,verbose=1,steps_per_epoch=50)
        # M1.save('AgeGenderModel.h5')

        print("================== EMOTION MODEL =====================")
        M2 = self.model_emotion(modal=modal)
        M2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        M2.fit(D2,epochs=100)

        
        M2.save('EmotionModel.h5')


        
        





            

        
    

