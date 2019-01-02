## 소스 출처 : https://hunkim.github.io/ml/	
## 아래 코드는 위 링크에서 학습한 머신 러닝 코드를 실습한 내용을 정리해 놓은 것으로, 
## 관련된 모든 내용은 원 링크를 참고해주시가 바랍니다

## Source Originated from : https://hunkim.github.io/ml/	
## The following code is a summary of the machine learning code learned from the link above.
## Please refer to the original link for all related contents.

# import tensorflow
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)	# tf.float32 타입으로 텐서 생성
node2 = tf.constant(4.0)				# 묵시적으로 tf.float32로 형변환이 이루어짐
node3 = tf.add(node1, node2)

# 각 노드를 출력하면 그 안의 값이 아닌 노드(텐서)의 타입 정보가 출력된다
print("node1:", node1, "node2:", node2)
print("node3:", node3)

# Create session
sess = tf.Session()

# 세션을 생성하고 세션으로 각 텐서를 실행시키면 텐서 안에 저장되어 있는 오퍼레이션이 실행된다.
print("sess.run(node1, node2): ", sess.run(node1, node2) )
print("sess.run(node3): ", sess.run(node3) )