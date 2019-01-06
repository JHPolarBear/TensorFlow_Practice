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
# 2차원 행렬로 데이터 셋 구성
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# X && Y Data
# Data가 행렬인 경우 shape에 유의하자
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 각 x_Data에 대응하는 W var
W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# H(x) 방정식 선언
hypothesis =tf.matmul(X,W) + b

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
								   feed_dict = {X: x_data, Y: y_data})

	if step%10 == 0:
		print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)