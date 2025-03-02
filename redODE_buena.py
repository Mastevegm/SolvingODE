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
###pensé que era arriba esta parte
#loss_tracker = keras.metrics.Mean(name="loss")

class ODEsolver(Sequential):
   loss_tracker = keras.metrics.Mean(name="loss")
   
   
   #def train_step(self, data):
      
    #  batch_size=tf.shape(data)[0]
      #x=tf.random.uniform(batch_size,1), minval=-5, maxval=5)
     #   with tf.GradientTape() as tape:
       # with tf.GradientTape() as Tape_a:
        #  Tape_a.watch(x)
         # y_pre=self(x, training=True)
        #dy = Tape_a.gradient(y_pre,x)
        #x_0=tf.zeros((batch_size,1))
        #y_0=self(x_0, training=True)
        #eq=x*dy+y_pre-x**2*keras.backend.cos(x)
        #ic=y_0
        #loss = keras.losses.mean_squared_error(0., eq)+keras.losses.mean_squared_error(0.,ic)
        
#usamos como base lo que se había visto en clase para la resolución de ODE
class ODEsolver(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            x_0 = tf.zeros((batch_size, 1))
            y_0 = self(x_0, training=True)
            eq = x * dy + y_pred - x ** 2 * keras.backend.cos(x)
            ic = y_0
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

        @property
        def metrics(self):
            return [keras.metrics.Mean(name='loss')]



#model = ODEdolver()
#Tuve varios errores porque estaba mal escrito
model = ODEsolver()
model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])
###Aquí debíamos poner el modelo, para que saque datos
tf.keras.layers.Dropout(.25, input_shape=(2,))


#No había colocado una y ni x
x=tf.linspace(-5,5,2000)
#No me salía por el history, que ponía un ajuste
history=model.fit(x, epochs=2000, verbose=1)
x_testv = tf.linspace(-5, 5, 2000)

y = [((x*np.sin(x))+(2*np.cos(x))-((2/x)*np.sin(x))) for x in x_testv]

a = model.predict(x_testv)

plt.grid()
plt.title('Red vs Analítica')

plt.plot(x_testv, a)
plt.plot(x_testv, y)
plt.show()
#exit()

model.save("red.h5")
#exit va debajo de model.save
exit()
#modelo_cargado = tf.keras.nodels.load_model('red.h5')
modelo_cargado = tf.keras.models.load_model('red5.h5')


