import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class DQN(Model):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv1 = Conv2D(32, 8, 4,activation='relu', padding='SAME')
        self.conv2 = Conv2D(64, 4, 2,activation='relu', padding='SAME')
        self.conv3 = Conv2D(64, 3, 1,activation='relu', padding='SAME')
        
        self.fc1 = Dense(512, 'relu')
        self.fc2 = Dense(n_actions)
        
    

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        

