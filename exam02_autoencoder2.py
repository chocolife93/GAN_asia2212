import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

# 이미지(데이터)를 줄였다 늘림 -> 오토인코더
input_img = Input(shape=(784,)) # 784 = 28 * 28 ; 28*28이미지를 flutten시킨것과 같음; 레이어가 아님
encoded = Dense(128, activation='relu')(input_img) # 덴스레이어 하나 덜렁 있음
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')

# 학습데이터 전처리
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1,784)
flatted_x_test = x_test.reshape(-1,784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs = 50, batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

decoded_img = autoencoder.predict(flatted_x_test[:10])

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