import tensorflow as tf
# placeHolder 란
# 마치 데이터베이스의 PreapreStatement처럼 질의문 ? 를 생각하면 된다.
# 사용자가 입력한값을 질의문에 대응시키기 위하여 ?를 대시하듯이
# 어떤 수식에 대응시키기 위한 변수의 틀을 미리 만들어 두는개념이다
# 예를 들어 다음의 수식을 보자
# a = [1,2,3]
# b = a*2
# b는 정해진 배열[1,2,3] * 2만 할줄 안다 마약 어떤요소의 배열이라도 연산시키고자 한다
# 위와 같이 값을 구체화 하지 않고 다만, 3개짜리 배열이다. 라고 틀을 만들어두면 어떤요소의 배열이라도 연산시킬 수 있다.

# 어떤 값이라도 담을 수 있는 정수형 3개을 담을 수 있는 placeHolder를 만들어요. placeHolder 명을 a라고 가정한다.
a = tf.placeholder(tf.int32,[3])
b = tf.constant(2)
x_op = a * b
# 수식을 실행시켜 / 먼저 tensor의 session을 얻어 와야한다.
sess = tf.Session()

# x_op를 실행하려한다. 결정되지않은값 플레이스홀더를 갖고 있기 때문에 반드시 값을 설정해주어야 한다.
r1 = sess.run(x_op, feed_dict={a:[1,2,3]})
print(r1)

r2 = sess.run(x_op, feed_dict={a:[5,3,1]})
print(r2)

row = [10,20,30]
r3 = sess.run(x_op,feed_dict={a:row})
print(r3)


