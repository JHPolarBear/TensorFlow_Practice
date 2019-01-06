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

xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype = np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_len = len(x_data[1])
y_len = len(y_data[1])

# X && Y Data
# Data가 행렬인 경우 shape에 유의하자
X = tf.placeholder(tf.float32, shape=[None, x_len])
Y = tf.placeholder(tf.float32, shape=[None, y_len])

# 각 x_Data에 대응하는 W var
W = tf.Variable(tf.random_normal([x_len, y_len]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# H(x) 방정식 선언
hypothesis =tf.sigmoid(tf.matmul(X,W) + b)

# Cost 방정식
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

#위의 cost minimization을 해주는 optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#Accuracy Computation

# hypothesis의 값이 0.5를 초과한 경우 1인 것으로 봄
predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)

#모든 predicted 값을 구한 후 평균을 내어 정확도 계싼
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float32 )	)

# Create session
with tf.Session() as sess:

	# variable 초기화
	sess.run(tf.global_variables_initializer())

	feed = {X: x_data, Y: y_data}

	# 학습 진행
	for step in range(10001):
		cost_val, _ = sess.run([cost, train],   feed_dict = feed)

		if step%500 == 0:
			#  500번째마다 정확도 체크
			a = sess.run([accuracy], feed_dict=feed)
			print(step, cost_val, a)

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
	print("\nhypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)