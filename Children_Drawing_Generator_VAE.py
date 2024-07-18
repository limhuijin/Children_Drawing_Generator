import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import mean, square, exp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from tensorflow.keras.layers import Layer, Input, Dense, Lambda

@tf.keras.utils.register_keras_serializable()
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # 재구성 손실과 KL 발산 추가
        reconstruction_loss = mean(binary_crossentropy(inputs, reconstructed)) * inputs.shape[1] * inputs.shape[2]
        kl_loss = -0.5 * mean(1 + z_log_var - square(z_mean) - exp(z_log_var), axis=-1)
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return reconstructed

# CSV 파일에서 이미지 경로 읽기
def load_image_paths(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    image_paths = [os.path.join(image_folder, f"{img_name}.png") for img_name in df['FileName']]  # 열 이름 수정
    return image_paths

# 이미지 로드 및 전처리 함수
def load_images(image_paths, image_size=(128, 128)):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(image_size)
        images.append(np.array(img))
    images = np.array(images).astype('float32') / 255.0  # 정규화
    return images

# VAE 모델의 인코더 부분
def build_encoder(latent_dim, input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(lambda t: t[0] + exp(t[1] / 2) * tf.random.normal(shape=(latent_dim,)))([z_mean, z_log_var])
    model = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model

# VAE 모델의 디코더 부분
def build_decoder(latent_dim, output_shape=(128, 128, 3)):
    model = tf.keras.Sequential([
        layers.Dense(16*16*128, activation='relu', input_shape=(latent_dim,)),
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same'),
        layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same'),
        layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
        layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')  # 3 채널 출력
    ])
    return model

# 결과 확인 함수
def plot_latent_space(vae, n=30, figsize=15, latent_dim=50):
    # 디스플레이 그리드 생성
    digit_size = 128
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.random.normal(size=(1, latent_dim))
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi
            x_decoded = vae.decoder(z_sample)
            digit = x_decoded[0].numpy().reshape(digit_size, digit_size, 3)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure)
    plt.show()

# 모델과 데이터 준비
latent_dim = 50
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')

# CSV 파일에서 이미지 경로 읽기 및 이미지 로드
csv_file = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/csv/Painting_Creativity_Tester_05.csv'  # CSV 파일 경로
image_folder = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/'  # 이미지 폴더 경로
image_paths = load_image_paths(csv_file, image_folder)
x_train = load_images(image_paths)

# 모델 훈련
vae.fit(x_train, epochs=100, batch_size=32)

# 모델 저장
vae.save('C:/Users/user/Desktop/coding/Children_Drawing_Generator/model/VAE/children_drawing_generator.keras')

# 결과 확인
plot_latent_space(vae, latent_dim=latent_dim)
