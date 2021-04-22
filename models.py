# ML models
from sklearn.decomposition import PCA

# Keras and Tensorflow
import keras

from tensorflow import keras, nn
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

'''VGG model'''
class VGG(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the VGG model
        vgg_conv = VGG16(weights='imagenet', input_shape=(224,224,3))
        self.VGG_model = Sequential()
        for layer in vgg_conv.layers[:-1]: # excluding last layer from copying
            self.VGG_model.add(layer)
                
    def freeze(self):
        self.VGG_model.trainable = False

    def predict(self, images):
        features = self.VGG_model.predict(images, verbose=1)

        return features



'''EfficientNet B3 model'''
class EffNetB3(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.EfficientNetB3_model = EfficientNetB3(weights='imagenet',
                                include_top = False, 
                                input_shape=(300, 300, 3),
                                pooling = 'avg')
                
    def freeze(self):
        self.EfficientNetB3_model.trainable = False

    def predict(self, images):
        features = self.EfficientNetB3_model.predict(images, verbose=1)

        return features



'''ResNet50 Model'''
class ResNet(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.ResNet_model = ResNet50(weights='imagenet', 
                                    input_shape=(512,512,3), 
                                    include_top = False, 
                                    pooling = 'avg')
                
    def freeze(self):
        self.ResNet_model.trainable = False

    def predict(self, images):
        features = self.ResNet_model.predict(images, verbose=1)

        return features



'''Keras model based IRMAE'''
class IRMAE(keras.Model):
    def __init__(self, mode, len_w = 4, latent_dim = 128):
        super().__init__()
        
        size = tuple()
        input_shape = tuple()
        if mode == "VGG":
            size = (8, 8, 64)
            input_shape = (64, 64, 1)
        elif mode == "Efficient":
            size = (4, 5, 64)
            input_shape =(32, 44, 1)
        else:
            size = (4, 8, 64)
            input_shape =(32, 64, 1)
        
        # encoder
        self.irmae = Sequential([
                                   keras.Input(shape = input_shape),
                                   Conv2D(8, (3, 3), activation='relu', padding='same'),
                                   MaxPooling2D(pool_size=(2, 2)),
                                   Conv2D(16, (3, 3), activation='relu', padding='same'),
                                   MaxPooling2D(pool_size=(2, 2)),
                                   Conv2D(32, (3, 3), activation='relu', padding='same'),
                                   MaxPooling2D(pool_size=(2, 2)),
                                   Conv2D(64, (3, 3), activation='relu', padding='same'),
                                   Flatten(),
                                   Dense(size[0]*size[1]*size[2], activation = nn.softmax)
                                  ])
        
        # W layer
        for i in range(len_w):	
            self.irmae.add(Dense(latent_dim, activation = nn.softmax))