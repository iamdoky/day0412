import tensorflow as tf
# tensorflow에서 사용할 상수[변하지 않는 값]
# 자바문법에 final int a = 120 같은 개념
a = tf.constant(120, name = 'a')
b = tf.constant(130, name = 'b')
c = tf.constant(140, name = 'c')

v = tf.Variable(0, name ='v')
calc_op = a + b + c

# 텐서 데이터 플로우 그래프에 연산을 계획하는 것이고
# 실제 계산은 session에 올려서 run을 시켜야 동작을 한다(계산을 한다.)

assingn_op = tf.assign(v, calc_op)

# 텐서의 수식을 실행 시키기 위하여 텐서의 세션이 필요하다.
sess = tf.Session()
sess.run(assingn_op)        # 연산한 결과를 텐서변수 v에 담는다. (assings 할때)
print(sess.run(v))