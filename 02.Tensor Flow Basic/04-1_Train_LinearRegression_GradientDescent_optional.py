## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

# H(x) = Wx + b (Linear Regression)
W = tf.Variable(5.)

# X && Y Data
X = [1,2,3]
Y = [1,2,3]

# H(x) 방정식 선언
hypothesis = X*W

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#직접 계산 한 그래디언트
gradient = tf.reduce_mean((W*X-Y)*X)*2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

#get Optimizer's gradient
gvs = optimizer.compute_gradients(cost)	# 위에서 정의한  cost function을 미분한 gradient
#gradient를 적용하여 minimize 진행
apply_gradients = optimizer.apply_gradients(gvs)

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(101):
	print(step, sess.run([gradient, W, gvs]))
	sess.run(apply_gradients)