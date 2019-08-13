import tensorflow as tf
mnist =tf.keras.datasets.mnist
(xtrain,y_train),(x_test,y_test)=mnist.load_data()
x_train =tf.keras.utils.normalize(xtrain)
x_test =tf.keras.utils.normalize(x_test)
import matplotlib.pyplot as plt
plt.imshow(x_train[0],cmap= plt.cm.binary)
plt.show()



model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(x_train,y_train,epochs=3)

val_loss,val_acc =model.evaluate(x_test,y_test)
model.save("mnist_model_tf.model")
new_mod=tf.keras.models.load_model('mnist_model_tf.model')

pred=new_mod.predict([x_test])
plt.imshow(x_test[1])
plt.show()
