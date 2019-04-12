import tensorflow as tf
import pandas as pd
import numpy as np

student = [['김경민',80,90,100],['박경민',70,80,90],['이경민',85,95,95],['진경민',87,97,77],['민경민',50,98,88]]

# palceholder를 이용하여 각 학생의 평균을 구하여 출력 하시오.
# ph_score = tf.placeholder(tf.int32,None)
# avg = tf.constant(3)
# result = ph_score/avg
# sess = tf.Session()
# for s in student:
#     score = s[1:]
#     name = s[:1]
#     s = sum(score)
#     test1 = sess.run(result, feed_dict={ph_score:s})
#     print(name[0]+'의 평균 점수',test1)

# 학생의 점수를 담을 3개짜리 배열을 placeHolder로 만든다.
ph_score = tf.placeholder(tf.int32,[3])
cnt = tf.constant(3)
# avg = (ph_score[0]+ph_score[1]+ph_score[2])/cnt
# avg = ph_score.mean()
avg = tf.reduce_mean(ph_score)
print(avg)

# 텐서를 실행시키기 위한 세션을 오픈
sess = tf.Session()

# 학생의 데이터 수 만큼 반복하여 run
for row in student:
    name, s = row[0], row[1:]
    r = ss = sess.run(avg,feed_dict={ph_score:s})
    print(name,r)
sess.close()



