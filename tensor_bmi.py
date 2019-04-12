import pandas as pd
import numpy as np
import tensorflow as tf
csv = pd.read_csv('bmi.csv')
# print(csv.head())

# 값 fat normal thin / 정답*label

# 학습하기 위한 Label의 종류 3가지를 one-hot Encoding : 이진화 하는거.

# 가장 큰 키와 가장 작은키
# print(csv['height'].max())
# print(csv['height'].min())

# 가장 큰 몸무게와 작은 몸무게
# print(csv['weight'].max())
# print(csv['weight'].min())

# 기계학습의 범위가 커서 feature들의 범위를 줄인다 이것을 정규화 라고 한다.
# 값의 범위를 축소하여 0과 1사이에 값들로 만든다
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100
# 가장 큰 키와 가장 작은키
# print(csv['height'].max(),csv['height'].min())

# 가장 큰 몸무게와 작은 몸무게
# print(csv['weight'].max(),csv['weight'].min())

# 답의 종류에 따른 one hot encoding 값을 정해요
bcalss = {"thin":[1,0,0],"normal":[0,1,0],"fat":[0,0,1]}
csv['label_fat'] = csv['label'].apply(lambda x: np.array(bcalss[x]))
print(csv.head())
# 원본 데이터의 label에 따라 one hot Encoding 테이블을 생성하여 새로운 칼럼 label_pat를 추가해 봅니다.

# 갖고있는 데이터를 모두 훈련에 참여시키면 훈련된 데이터만 잘 알아 맞추,, 새로운 데이터에 대해서는 정확도가 떠렁짐 = overfit
# 기준이 애매하다... 보통 갖고 있는 데이터에 70 80프로를 훈련에 참여시키고 나머지에 데이터로 검증을 해야한다.

# 전체 데이터 수는 2만여건으로 이중에 3분의2를 훈련데이토 3분의1을 검증데이터로 사용하려한다.
# 15000번째 인덱스부터 2만번 까지의 데이터를 뽑아 검증데이토ㅗ 담아요
test_csv = csv[15000:20000]

# 검증을 위한 데이터 test_csv로 문제와 답을 나눈다.
test_pat = test_csv[['weight','height']]
test_ans = list(test_csv['label_fat'])
# print(test_pat.head())
# print(test_ans.head())

# 훈련을 시키기 위한 문제를 위한 placeholder 를 만든다.
x = tf.placeholder(tf.float32,[None,2])
# 훈련을 시키기 위한 답 placeholder를 만든다.
y_ = tf.placeholder(tf.float32,[None,3])

# 가중치를 위한 배열을 만들어요.
# [feature(키,무게)의 수, 답(label(fat / normal / thin)의 수]
# 변수선언
W = tf.Variable(tf.zeros([2,3]))    # 가중치
# 답의 수가 3가지, 즉 fat thin normal 각각의 feature의 각각 weright을 적용한 thin/fat/normal이 되려면 얼마를 넘어야 한다.
# 각각으ㅔ 대한 기준치 (임계치)가 3개의 값이 필요하다.
# 일단 0으로 채워두면 tensor가 학승을 하면서 이것을 알맞은 값을 셋팅응ㄹ 합니다.
b = tf.Variable(tf.zeros([3]))      # 편향    바이어스의 값의 수는 답을 수와 동일하게 한다.
# 텐서가 제공하는 기계학습을 위한 softmax 객체를 생성한다. ==> 모델을 만든다.
# softmax 회귀 정의 matmul.. y = wx + b 의 함수를 만든다,...?
y = tf.nn.softmax(tf.matmul(x,W)+b)     # 경사하강법...?              # c

# 훈련된 결과와 진짜 답과의 거리를 가능하면 작게 만들기 위한 객체를 생성한다.

# 진짜 답과 예측한 답의 합을 담느다.(잔차의 합)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))                      # B
# 그 잔차의 합의 최소화 되게 해주는 개체를 생성한다.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)                           # A
# 이 객체를 텐서를 이용하여 실행한다. 이수식에는 2개의 플레이스홀더가 있다.

# A의 수식에 사용된 cross entropy는 B이며 B의 cross _entrpy는 y_와ㅛfmf vhgkagksek.
# y_는 진짜 답을 담는 플레이스 홀더이다.
# y는 훈려을 위한 식이며 즉 훈련된 결과과 담기는 변수. 현재 식에 문제가 담길 placeholder x가 포함된다.
# tensor에서 실행시킬때 y_와 x에 해당하는 값을 설정해야 한다.

# 훈련시키는 과정에서 정확도를 확인하기 위한 수식

# 예측한 답과 진짜 답을 비교하여 predict에 담는다.
predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 예측 답과 진짜답을 비교한 predict는 boolean 배열을 실수형으로 변환하여 평균값을 계산.
accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 원본 데이터 2만개 중에 검증용 데이터 15000~20000건을 제외ㅘㄴ 모든 데이터를 훈련시킨다.
# 15000개를 한번에 훈련 시키지 않고 100개씩 끊어서 훈려한데,, why?

# i는 0,100,200~ 14900
# i번째 부터 100개씪 봅아온다. rows에 담는다.
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1+i : 1+i+100]
    x_pat = rows[['weight','height']]
    y_ans = list(rows['label_fat'])
    fd = {x:x_pat, y_:y_ans}
    sess.run(train,feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy,feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x:test_pat,y_:test_ans})
        print("step= ",step,"cre",cre,"acc=",acc)

acc = sess.run(accuracy,feed_dict={x:test_pat, y_:test_ans})
print("정답률 =", acc)

# for step in range(0,14900,100):
#     print(step)

sess.close()







