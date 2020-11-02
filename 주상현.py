import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


tf.compat.v1.reset_default_graph()

idx2char = ['_','I','l','o','v','e','y','u','!']

# hihell을 입력하여 ihello를 도출한다. -> 자기 자신의 다음 알파벳을 도출

x_data = [[0,1,0,2,3,4,5,0,6,3,7]]  # _I_love_you
x_one_hot = [[[1,0,0,0,0,0,0,0,0],  # _ 0
             [0,1,0,0,0,0,0,0,0],   # I 1
             [1,0,0,0,0,0,0,0,0],   # _ 0
             [0,0,1,0,0,0,0,0,0],   # l 2
             [0,0,0,1,0,0,0,0,0],   # o 3
             [0,0,0,0,1,0,0,0,0],   # v 4
             [0,0,0,0,0,1,0,0,0],   # e 5
             [1,0,0,0,0,0,0,0,0],   # _ 0
             [0,0,0,0,0,0,1,0,0],   # y 6
             [0,0,0,1,0,0,0,0,0],   # o 3
             [0,0,0,0,0,0,0,1,0]]]  # u 7

y_data = [[1,0,2,3,4,5,0,6,3,7,8]] #ihello

num_classes = 9
input_dim = 9
hidden_size = 9
batch_size = 1
sequence_length = 11
learning_rate = 0.1

#sequence_length = 글자의 개수(6)를 의미하고, input_dim = 백터 인덱스 값 (5)
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label


#num_units -> 행렬곱의 가능을 위해 input dimension과 같다고 생각하면 됌
cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)

#처음 노드의 가중치를 위한 값을 0으로 하겠다
initial_state = cell.zero_state(batch_size, tf.float32)

#dynamic_rnn은 셀의 모양을 다양하게 해주는 것 
#엄밀하게 X의 데이터에 대해서 자동으로 시퀀스를 결정해주는 것
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state = initial_state, dtype = tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])

outputs = tf.contrib.layers.fully_connected(
    inputs = X_for_fc, num_outputs = num_classes, activation_fn = None)


#batch_size = 'hihello' 하나, sequence_length = 'hihell' / 'ihello' -> 6
#num_classes = 글자 하나당 벡터 5개 [0,1,0,0,0]
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size,sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = outputs, targets = Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        #squeeze -> 차원에서 1인 것을 없애줌
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))