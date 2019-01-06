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

# Load data
xy = np.loadtxt('data-04-zoo.csv', delimiter = ',', dtype = np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_len = len(x_data[1])
nb_classes = 7

print(x_len, nb_classes)

# X && Y Data
# Data가 행렬인 경우 shape에 유의하자
X = tf.placeholder(tf.float32, shape=[None, x_len])

Y = tf.placeholder(tf.int32, shape=[None, 1])   #0~6 사이 하나의 값을 갖게 됨 shape(?,1)
Y_one_hot = tf.one_hot(Y, nb_classes)           #클래스(레이블) 갯수만큼의 one_hot 행렬을 생성 shape(?,1,7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) #one_hot 함수가 불필요한 axis를 추가로 만들어 이를 제거 shape(?,7)

# 각 x_Data에 대응하는 W var
W = tf.Variable(tf.random_normal([x_len, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

# H(x) 방정식 선언
logits = tf.matmul(X,W) + b
hypothesis =tf.nn.softmax(logits)

# Cost 방정식(Cross entropy) - tf에서 제공되는 버전
# logit과 one hot(위 식의 Y)를 받아 -tf.reduce_sum(Y*tf.log(hypothesis), axis=1) 를 실행
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

#위의 cost minimization을 해주는 optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# 정확도 체크
prediction = tf.argmax(hypothesis, 1)       #현재 hypohtesis에서 나온 값의 클래스 인덱스
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # 실제 인덱스
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 위 2개를 비교하여 일치하는 확률 구함

# Create session
with tf.Session() as sess:

	# variable 초기화
	sess.run(tf.global_variables_initializer())

	feed = {X: x_data, Y: y_data}
	# 학습 진행
	for step in range(2001):
		sess.run(optimizer,   feed_dict = feed)

		if step%100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict=feed)
			print("stdp: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
			
	pred = sess.run(prediction, feed_dict = {X: x_data})
	
	for p, y in zip(pred, y_data.flatten()):
		print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

