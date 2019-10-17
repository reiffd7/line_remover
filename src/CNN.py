from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io, color, filters, feature, restoration
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.metrics import classification_report, confusion_matrix
import csv
import numpy as np
import glob
import os
import sys
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  
sys.path.append(ROOT_DIRECTORY)


class imageCNN(object):

    def __init__(self, train_path, test_path, holdout_path, model_name):
        self.train_path = train_path
        self.test_path = test_path
        self.holdout_path = holdout_path
        self.model_name = model_name
        self.len_init()

    def len_init(self):
        self.n_train = sum(len(files) for _, _, files in os.walk(self.train_path))  # number of training samples
        self.n_val = sum(len(files) for _, _, files in os.walk(self.holdout_path))  # number of validation samples
        self.n_test = sum(len(files) for _, _, files in os.walk(self.test_path)) # number of test samples
        self.nb_classes = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_path))


    def param_init(self, epochs, batch_size, image_size, base_filters, final_layer_neurons):
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nb_filters = base_filters
        self.neurons = final_layer_neurons

    def load_and_featurize_data(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        holdout_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
                        self.train_path,
                        target_size=(30, 30),
                        batch_size=1,
                        class_mode='binary',
                        shuffle=True)
        
        self.test_generator = test_datagen.flow_from_directory(
                        self.test_path,
                        target_size=(30, 30),
                        batch_size=1,
                        class_mode='binary',
                        shuffle=False)

        self.holdout_generator = holdout_datagen.flow_from_directory(
                        self.holdout_path,
                        target_size=(30, 30),
                        batch_size=1,
                        class_mode='binary',
                        shuffle=False)


    def define_model(self, kernel_size=(3, 3), pool_size=(2, 2), dropout=0.25, num_blocks=1):

        print('Beginning model defining process... ')
        self.model = Sequential()
        self.model.add(Conv2D(64, (4, 4), input_shape=(self.image_size[0], self.image_size[1], 3),
                                padding='valid',
                                name='Convolution-1',
                                activation='relu'))
        print('First layer added')
        self.model.add(Conv2D(32, (4, 4), padding='valid',
                                name='Convolution-2',
                                activation='relu'))
        print('Second layer added')
        self.model.add(MaxPooling2D(pool_size=(4, 4),
                                    name='Pooling-1'))
        print('Pooling :)')
        self.model.add(Conv2D(32, (2, 2), padding='valid',
                                name='Convolution-3',
                                activation='relu'))
        print('Thrid layer added')
        self.model.add(Conv2D(64, (2, 2), padding='valid',
                                name='Convolution-4',
                                activation='relu'))
        print('Fourth layer added')
        self.model.add(MaxPooling2D(pool_size=pool_size,
                                    name='Pooling-2'))
        print('Pooling :)')
        self.model.add(Dropout(dropout))
        print('Dropping ', dropout)
        self.model.add(Flatten())
        print('Model flattened out to ', self.model.output_shape)
        self.model.add(Dense(self.neurons, name='Dense-1', activation='relu'))
        self.model.add(Dense(1, name='Dense-2', activation='sigmoid'))
        print('Compiling... ')

        self.model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy', Precision(), Recall()])
        print('Done')



    def train_model(self):
        
        self.hist = self.model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.n_train/self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.holdout_generator,
                validation_steps=self.n_val/self.batch_size,    
                use_multiprocessing=True)
       
    def evaluate_model(self):
        self.metrics = self.model.evaluate_generator(self.test_generator,
                                           steps=self.n_test/self.batch_size,
                                           use_multiprocessing=True,
                                           verbose=1)
        

    def save_history(self):
        hist_path = os.path.join(MODEL_DIRECTORY, 'model_history/{}.csv'.format(self.model_name))
        with open(hist_path, 'w') as csv_file:
            my_dict = self.hist.history
            w = csv.DictWriter(csv_file, my_dict.keys())
            w.writeheader()
            for i in range(0, len(list(my_dict.values())[0])):
                w.writerow({list(my_dict.keys())[0]: list(my_dict.values())[0][i],
                           list(my_dict.keys())[1]: list(my_dict.values())[1][i],
                           list(my_dict.keys())[2]: list(my_dict.values())[2][i],
                           list(my_dict.keys())[3]: list(my_dict.values())[3][i],
                           list(my_dict.keys())[4]: list(my_dict.values())[4][i],
                           list(my_dict.keys())[5]: list(my_dict.values())[5][i],
                           list(my_dict.keys())[6]: list(my_dict.values())[6][i],
                           list(my_dict.keys())[7]: list(my_dict.values())[7][i],})
        print("Saved model and metrics to disk")
       
    
       

    def save_model(self):
        # Save model and weights
        model_path = hist_path = os.path.join(MODEL_DIRECTORY, 'model_names/{}.h5'.format(self.model_name))
        self.model.save(model_path)
        print("Saved model to \"" + model_path + "\"")


    # def save_model_predictions(self):
    #     self.Y_pred = self.model.predict_generator(self.test_generator, 
    #                                     steps=self.n_test/self.batch_size,
    #                                     use_multiprocessing=True, 
    #                                     verbose=1)
    #     self.y_pred = np.argmax(self.Y_pred, axis=1)
    #     cm = confusion_matrix(self.test_generator.classes, self.y_pred)
    #     print(cm)


    # def evaluate_model(self):
    #     # Score trained model.
    #     scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
    #     print('Test loss:', scores[0])
    #     print('Test accuracy:', scores[1])


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # train_lines = glob.glob('../data/output/train/lines/*')
    # train_drawings = glob.glob('../data/output/train/drawings/*')


    train_path = os.path.join(ROOT_DIRECTORY, 'data/output/train')  
    test_path = os.path.join(ROOT_DIRECTORY, 'data/output/test')
    holdout_path = os.path.join(ROOT_DIRECTORY, 'data/output/val')  

    print('Creating class')

    cnn = imageCNN(train_path, test_path, holdout_path, 'First')

    print("Initializing Parameters")
    cnn.param_init(
        epochs=5, 
        batch_size=32, 
        image_size=(30, 30), 
        base_filters=16, 
        final_layer_neurons=128)


    print("Loading and featurizing the data")

    cnn.load_and_featurize_data()

    cnn.define_model()

    print("Training the model...")
    cnn.train_model()

    cnn.evaluate_model()
    cnn.save_history()
    cnn.save_model()
    # cnn.save_model_predictions()
