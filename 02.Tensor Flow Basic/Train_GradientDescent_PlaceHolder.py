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
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# X && Y Data
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) 방정식 선언
hypothesis = X*W + b

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Optimizer 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(2001):
	cost_val, W_val, b_val, _ = \
		sess.run( [cost, W, b, train], 
			feed_dict={ X:[1,2,3], Y:[1,2,3]} )
	
	#Create Log at every 20th step	
	if step % 20 == 0 :
		print (step, cost_val, W_val, b_val)
		
# 학습된 모델 테스트
print(	sess.run(hypothesis, feed_dict = {X:[5]})	)
print(	sess.run(hypothesis, feed_dict = {X:[2.5, 3.7]})	)