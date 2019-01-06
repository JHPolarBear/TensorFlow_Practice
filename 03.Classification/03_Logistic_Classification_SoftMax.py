## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import module
import tensorflow as tf

import matplotlib as plt

import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_len = len(x_data[1])
nb_classes = len(y_data[1])

# X && Y Data
# Data가 행렬인 경우 shape에 유의하자
X = tf.placeholder(tf.float32, shape=[None, x_len])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

# 각 x_Data에 대응하는 W var
W = tf.Variable(tf.random_normal([x_len, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

# H(x) 방정식 선언
hypothesis =tf.nn.softmax(tf.matmul(X,W) + b)

# Cost 방정식
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

#위의 cost minimization을 해주는 optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# Create session
with tf.Session() as sess:

	# variable 초기화
	sess.run(tf.global_variables_initializer())

	feed = {X: x_data, Y: y_data}

	# 학습 진행
	for step in range(10001):
		sess.run(optimizer,   feed_dict = feed)

		if step%500 == 0:
			#  500번째마다 정확도 체크
			cost_val = sess.run([cost], feed_dict=feed)
			print(step, cost_val)

	# accuracy
	a = sess.run(hypothesis, feed_dict = {X:[[1,11,7,9],
											 [1,3,4,3],
											 [1,1,0,1]]})
	print(a, sess.run(tf.arg_max(a, 1)))