## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

# X && Y Data
x_train = [1,2,3]
y_train = [1,2,3]

# H(x) = Wx + b (Linear Regression)
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# H(x) 방정식 선언
hypothesis = x_train*W + b

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Optimizer 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행

for step in range(2001):
	sess.run(train)		# Cost 최소값을 찾는 train 반복 실행된다
	
	#Create Log at every 20th step	
	if step % 20 == 0 :
		print (step, sess.run(cost), sess.run(W), sess.run(b))