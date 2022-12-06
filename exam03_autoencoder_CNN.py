import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
## 모델디자인-----------------------------------------
# encoder

input_img = Input(shape = (28,28,1)) # 1은 원컬러(=흑백)라는 뜻
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2,2), padding='same')(x) # 맥스풀 커널은 겹치지 않게 지나간다
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPool2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPool2D((2,2), padding='same')(x)

# decoder
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 출력값이 0~1값이어야 하므로 활성화함수는 시그모이드


autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 입력이미지
(x_train,_), (x_test, _) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
# print(conv_x_train.shape)
# print(conv_x_train
# exit()

# 데이터에 잡음(노이즈) 섞기
noise_factor = 0.5 # 노이즈 스케일; 잡음의 크기가 0.5 ; 잡음의 강도를 조절하기 위함
conv_x_train_nosiy = conv_x_train + np.random.normal(0, 1, size=conv_x_train.shape) * noise_factor
conv_x_train_nosiy = np.clip(conv_x_train, 0.0, 1.0) # 0.0보다 낮은것은 0.0이되고  1.0보다 높은것은 1.0이 됨
conv_x_test_nosiy = conv_x_test + np.random.normal(0, 1, size=conv_x_test.shape) * noise_factor
conv_x_test_nosiy = np.clip(conv_x_test, 0.0, 1.0) # 0.0보다 낮은것은 0.0이되고  1.0보다 높은것은 1.0이 됨
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, 10, i +1)
    plt.imshow(conv_x_test_nosiy[i].reshape(28,28)) # 입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,10,i+1+n)
    plt.imshow(conv_x_test_nosiy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


fit_hist = autoencoder.fit(conv_x_train_nosiy, conv_x_train, epochs=50, batch_size=256, validation_data=(conv_x_test_nosiy,conv_x_test)) # 여기 질문 : 매소드 넣는 순서
autoencoder.save('./models/autoencoder_noisy.h5')

decoded_img = autoencoder.predict(conv_x_test[:10])


plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, 10, i +1)
    plt.imshow(conv_x_test_nosiy[i].reshape(28,28)) # 입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,10,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.plot(fit_hist.history(['loss']))
plt.plot(fit_hist.history(['val_loss']))
plt.show() # 압축하는 과정에서 손실이 있었으니 복원이


'''
잡음제거를 어디에쓰냐? 
ex) 카메라 - 밤에 찍은 사진 -> 선명하게 볼까? -> 학습된 이미지 중에(만) 복원함 ; 새로운 이미지는 못만들어 냄
인코딩 -> 이미지의 특성을 압축하여 저장 ==> 디코더 -> 특성 기반으로 이미지를 만들어 냄

gan은 encoder부분이 없음; encoder가 없으므로 데이터가 없음. 그래서 잡음을 넣고 새로운 이미지를 만듦
'''