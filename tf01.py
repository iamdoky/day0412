import tensorflow as tf

a = tf.constant(2)      # a는 2이외의 다른값을 가질 수 없다. ==> 상수 만들기
b = tf.constant(3)
c = tf.constant(4)

calc1_op = a + b * c
# print(calc1_op)     # tensor의 상수나 변수 수식의 결과를 바로 출력할 수 없다.
                      # tensor의 환경에서 실행시키고 확인할 수 있다.
calc2_op = (a+b) * c

sess = tf.Session()
r1 = sess.run(calc1_op)
r2 = sess.run(calc2_op)

print(r1)
print(r2)

