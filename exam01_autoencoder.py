import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

# 이미지(데이터)를 줄였다 늘림 -> 오토인코더
input_img = Input(shape=(784,)) # 784 = 28 * 28 ; 28*28이미지를 flutten시킨것과 같음; 레이어가 아님
encoded = Dense(32, activation='relu') # 덴스레이어 하나 덜렁 있음
encoded = encoded(input_img) # 덴스레이어를 거친 인풋이미지
decoded = Dense(784, activation='sigmoid') # 덴스레이어 하나 덜렁 있음
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# encoder : 신호를 코드로 바꿈,코드를 만듦 ; decoder : 코드를 신호로 바꿈 ;; 신호 = 봉화 ;;코드로 바꾸는 이유 : 통신매체
'''
코덱 : enCODer + DECoder

오토인코더
'''

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,)) # 인풋은 레이어가 아니고 그냥 입력되는 공간
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')

# 학습데이터 전처리
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

#지도학습 : 타겟(답)이 있음, 라벨(Y)이 있음
#비지도학습 :라벨이 없음
# 오토인코더는 비지도학습 (or 자기지도학습)

flatted_x_train = x_train.reshape(-1,784)
flatted_x_test = x_test.reshape(-1,784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs = 50, batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10

plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10,i+1) # 첫번째 줄 ,
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n) # decoded img
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()