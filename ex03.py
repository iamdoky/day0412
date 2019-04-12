# 크기가 상관없는 정수형 1차원 배열의 placeholder 만들고 이것의 *2 한 결과를 tensor를 이용하여 출력

import tensorflow as tf

arr = tf.placeholder(tf.int32,None)
num = tf.constant(5)
result = arr * num
sess = tf.Session()

t = [5,3,2,1,2,3,1,5,1,1]

re1 = sess.run(result,feed_dict={arr:[5,3,2,6,1]})
re2 = sess.run(result,feed_dict={arr:[5,3,2,6,1,4,3,2,1,5,2,1,3,2,3,1,5,3,6,7,8,7,500]})
re3 = sess.run(result,feed_dict={arr:t})

print(re1)
print(re2)
print(re3)

