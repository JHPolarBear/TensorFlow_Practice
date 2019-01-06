## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

#import matplotlib
import matplotlib.pyplot as plt

# H(x) = Wx + b (Linear Regression)
W = tf.placeholder(tf.float32)

#이번 예제에서는 사용 안함
#b = tf.Variable(tf.random_normal([1]), name = 'bias')

# X && Y Data
X = [1,2,3]
Y = [1,2,3]

# H(x) 방정식 선언
hypothesis = X*W

# Cost 방정식
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Create session
sess = tf.Session()

# variable 초기화
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

# 학습 진행
for i in range(-30, 50):
	feed_W = i*0.1
	curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W})
	W_val.append(curr_W)
	cost_val.append(curr_cost)

# 그래프 출력
plt.plot(W_val, cost_val)
plt.show()