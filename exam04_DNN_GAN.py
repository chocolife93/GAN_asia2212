# https://ebbnflow.tistory.com/119          :           [인공지능] ANN(인공신경망), DNN(딥뉴럴네트워크), CNN(합성곱신경망), RNN(순환신경망) 개념과 차이

import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
import os

OUT_DIR = './DNN_out'
img_shape = (28, 28, 1)
epochs = 100000 # 학습의 횟수
batch_size = 128 # 딥러닝에서 배치는 모델의 가중치를 한번 업데이트시킬 때 사용되는 샘플들의 묶음을 의미;
noise = 100
sample_interval = 100
mnist_data = mnist.load_data()
print(mnist_data)
(x_train, _), (_, _) =mnist.load_data()
print(x_train)
print(x_train.shape)
print('----------------------------------------------------')
# 전처리
# 스케일링
x_train = x_train / 127.5 - 1 # -1~1사이 값으로 변환
x_train = np.expand_dims(x_train, axis=3) # np.expand_dims() : 차원을 늘려라 # 공부 : numpy array
print(x_train.shape)
## axis : 차원의 축; shape의 각 요소위치는 차원의 축을 말하고 x.shape >> (1,2,3,4)일 때, axis=0은 1의 위치, axis =3은 4의 위치이다

## 모델만들기
# generator
generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01)) # 렐루와 릭키렐루의 차이 # alpha = 음수값에서의 기울기, 주로 0.01사용 # 이 모델에서는 찐한부분에서 강하게 반응하기 위해 사용; 활성화함수를 레이어 생성 코드에 안넣은 이유는 리키렐루의 알파값을 넣어주기 위해 따로 뺌
generator.add(Dense(784, activation='tanh')) # 여기서 노드수가 784가 된 이유 공부; 활성화 함수 하이퍼볼릭탄젠트 공부
generator.add(Reshape(img_shape)) # 출력이 28*28*1 인 이미지
generator.summary() # 제너레이터만 가지고는 로스를 볼 수 없음, 제너레이터만가지고는 학습을 할 수 없음, 뒤에 이진분류기를 같이 붙여 같이 학습을 시켜야함, 이것이 gan모델
'''
# 이진분류기
lrelu = LeakyReLU(alpha=0.01) # 리키렐루의 디폴트는 0.3
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu)) # 리키렐루를 레이어 코드에 넣으려면 미리 리키렐루의 알파값을 지정한 변수를 넣으면 된다.
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# GAN모델 = generator + 이진분류기
gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy',optimizer='adam') # 간모델은 generator가 만들어낸 이미지를 진짜다라고 하게 만들어야함

# 타겟설정
real = np.ones((batch_size, 1)) # np.ones(쉐입모양) : 1로채워진 행렬을 만들어줌
print(real)
fake = np.zeros((batch_size,1)) # np.zeros(쉐입모양) : 0으로채워진 행렬을 만들어줌
print(fake)

# 모델학습(generator + discriminator)
for epoch in range(epochs):
    idx = np.random.randint(0,x_train.shape[0], batch_size) #
    real_imgs = x_train[idx]

    # 가짜이미지 : 제너레이터가 만들어낸 이미지
    z = np.random.normal(0,1,(batch_size, noise)) # 100개씩 128개 묶여있는 잡음을 만들어 냄
    fake_imgs = generator.predict(z) #

    d_hist_real = discriminator.train_on_batch(real_imgs, real) # fit : 에폭스를 줌; train : 이미지(데이터 : (입력데이터, 출력데이터))를 주면 이거만 학습하고 끝냄 = 1에폭하고 끝냄
    d_hist_real = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = np.add(d_hist_real, d_hist_real) * 0.5 # 평균을 냄 # d_hist_real, d_hist_real이 어레이이기 때문에 합칠 수 있음

    discriminator.trainable = False # discriminator가 학습이 안되게 함 # 후에 간모델을 학습할 때 discriminator가 한번더 학습을 하게되면 성능이 너무 좋아져 generator가 학습이 안됨 그래서 학습을 막아줌

    # GAN모델 학습
    # discriminator가 학습이 안됨 => discriminator 추가 학습
    if epoch % 2 == 0:
        z = np.random.normal(0,1, (batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real) # 제너레이터가 만들어낸 이미지(잡음)가 진짜라고 학습시킴

    # 이미지 저장
    # 10만번 다 저장하는건 너무 많으니까 100번 학습할 때마다 저장
    if epoch % sample_interval == 0:
        print('%d, [loss of discriminator: %f, acc.: %.2f%%], [loss of GAN : %f]'%(epoch, d_loss, d_loss, gan_hist))
        row = col = 4
        z = np.random.normal(0,1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5 # 0에서 1사이의 값으로 변환
        _, axs = plt.subplots(row, col, figsize=(5, 5), sharey = True, sharex = True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1)) #path 경로를 합침
        plt.savefig(path)
        plt.close()
'''
''' 참조
https://yeomko.tistory.com/39       :   활성화함수의 종류
https://bskyvision.com/entry/%EB%B0%B0%EC%B9%98batch%EC%99%80-%EC%97%90%ED%8F%AC%ED%81%ACepoch%EB%9E%80 :   batch_size & epoch
'''

''' 공부할 내용
Dense layer : 추출된 정보들을 하나의 레이어로 모으고, 우리가 원하는 차원으로 축소시켜서 표현하기 위한 레이어; 다중분류기 - 활성화함수 : softmax, 이진분류기 - 활성화함수 : sigmoid
배치사이즈 
에폭

'''


# 수업해본 모델이 (DNN)이 학습이 잘 안됨 =>DNN은 몇번 학습 하면 1번 되는데 CNN으로 하면 확실히 학습이 될 것임