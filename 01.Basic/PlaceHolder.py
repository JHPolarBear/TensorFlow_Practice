## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

# 실제 런 타임에 값을 지정할 수 있는 변수 역활을 하는 듯
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b	# +를 사용하면 tf.add(a+b) 와 동일한 코드가 생성된다.

# Create session
sess = tf.Session()

# run the operation
print(sess.run(adder_node, feed_dict = { a:3, b:4.5 } )	)
print(sess.run(adder_node, feed_dict = { a:[1,3], b:[2,4] }	)	)