# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

def display(char):
    c = [' ','.','-','*','=','+','8','0']
    print(''.join(["-" for x in range(len(char[0]))]))
    for row in char:
        for pixel in row:
            print(c[(pixel)//32],end='')
        print()
    print(''.join(["-" for x in range(len(char[0]))]))
    
print("TensorFlow Version:",tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data:",len(x_train),"records")
print("Testing data:",len(x_test),"records")

x_train_float, x_test_float = x_train / 255.0, x_test / 255.0

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# no dropout
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
                            ])
model1.compile(loss=loss_fn,metrics=['accuracy'])

# dropout
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])
model2.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

test_loss_1, test_acc_1 = model1.evaluate(x_test_float, y_test, verbose=2)
test_loss_2, test_acc_2 = model2.evaluate(x_test_float,y_test,verbose=2)

prob1 = tf.keras.Sequential([model1,tf.keras.layers.Softmax()])
prob2 = tf.keras.Sequential([model2,tf.keras.layers.Softmax()])

