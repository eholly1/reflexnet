import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

import network

N = 10000
batch_size = 64
input_size = 20
output_size = 10

start = time.time()

input = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, input_size])

hidden = Dense(300, input_shape=[None, input_size], activation='relu')
output = Dense(output_size, activation='linear')
hidden_vals = hidden(input)
ff = output(hidden_vals)

subnets = []
for _ in range(60):
    hidden = Dense(5, input_shape=[None, input_size], activation='relu')
    output = Dense(output_size, activation='linear')
    subnets.append(output(hidden(input)))
subnet = tf.add_n(subnets)

time_ff = 0.0
time_subnets = 0.0

time_setup = time.time() - start

with tf.compat.v1.Session().as_default():
    tf.compat.v1.initializers.global_variables().run()

    for _ in range(N):
        input_val = np.random.uniform(size=[batch_size, input_size])
        start = time.time()
        ff.eval(feed_dict={
            input: input_val
        })
        time_ff += time.time() - start

        start = time.time()
        subnet.eval(feed_dict={
            input: input_val
        })
        time_subnets += time.time() - start

    print()
    print()
    print("FF time: ", time_ff)
    print("Subnet time: ", time_subnets)