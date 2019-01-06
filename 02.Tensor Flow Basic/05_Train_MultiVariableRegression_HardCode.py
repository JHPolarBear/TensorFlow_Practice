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

# Data set
# 이렇게 작성하는 방식은 데이터가 많아질수록 비효율적이기 때문에 실제로는 이렇게 안한다
# 기본 원리를 확인한기 위한 코드로 참고만 합시다
x1_data = [73.,93.,89., 96., 73.]
x2_data = [80.,88.,91., 98., 66.]
x3_data = [75.,93.,90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# X && Y Data
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 각 x_Data에 대응하는 W var
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# H(x) 방정식 선언
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#위의 cost minimization을 해주는 optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
								   feed_dict = {x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})

	if step%10 == 0:
		print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)