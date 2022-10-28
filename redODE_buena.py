###Ingresamos las paqueterías###
import tensorflow as tf
import matplotlib
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

loss_tracker = keras.metrics.Mean(name="loss")

class ODEsolver(Sequential):
   def train_step(self, data):
      x=tf.random.uniform((95,1), minval=-5, maxval=5)
      with tf.GradientTape() as tape:
        with tf.GradientTape() as Tape_a:
          Tape_a.watch(x)
          y_pre=self(x, training=True)
        dy = Tape_a.gradient(y_pre,x)
        x_0=tf.zeros((95,1))
        y_0=self(x_0, training=True)
        eq=x*dy+y_pre-x**2*keras.backend.cos(x)
        ic=y_0
        loss = keras.losses.mean_squared_error(0., eq)+keras.losses.mean_squared_error(0.,ic)
        
        
    trainable_vars=self.trainable_variables
    gradients=tape.gradient(loss,trainable_vars)
    self.optimizer.apply_gradients(loss, trainable_vars)
    loss_tracker.update_state(loss)
    return {m.name: m.result() for m in self.metrics}
   
    
   @property
   def metrics(self):
      return [loss_tracker]

model = ODEdolver()

model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])


x=tf.linspace(-5,5,100)
a=model.predict(x_testv)
plt.plot(x_testv, a)
plt.plot(x_testv, np.exp(-x*x))
plt.show()
exit()

model.save("red.h5")

modelo_cargado = tf.keras.nodels.load_model('red.h5')



