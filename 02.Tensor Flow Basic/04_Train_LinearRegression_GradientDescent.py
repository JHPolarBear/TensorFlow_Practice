## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

# H(x) = Wx + b (Linear Regression)
W = tf.Variable(tf.random_normal([1]), name = 'weight')

# X && Y Data
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

x_data = [1,2,3]
y_data = [1,2,3]

# H(x) 방정식 선언
hypothesis = X*W

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize function (optimizer로 하던걸 직접 구현)
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W - learning_rate*gradient
update = W.assign(descent)

#위의 cost minimization을 해주는 optimizer
#지금이야 식이 간단하니 직접 미분을 했지만 실제로는 아래 함수를 사용합시다.
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
#train = optimizer.minimize(cost)

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(21):
	sess.run( update, feed_dict = {X: x_data, Y: y_data})
	
	#Create Log at every 20th step
	print (step, sess.run(cost, feed_dict = {X: x_data, Y: y_data}), sess.run(W))