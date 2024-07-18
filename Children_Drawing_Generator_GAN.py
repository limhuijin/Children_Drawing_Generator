import os
import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import pandas as pd

# 데이터 로드 및 전처리
csv_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/csv/Painting_Creativity_Tester_05.csv'
data = pd.read_csv(csv_path)

image_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/'
image_paths = [os.path.join(image_dir, f"{file_name}.png") for file_name in data['FileName']]

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)  # 이미지를 RGB로 읽음
    image = tf.image.resize(image, [64, 64])  # 이미지 크기를 64x64로 조정
    image = (image - 127.5) / 127.5  # 이미지를 [-1, 1] 범위로 정규화
    return image

image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
image_dataset = image_dataset.map(load_and_preprocess_image).batch(32)  # 배치 크기 설정

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        Input(shape=(latent_dim,)),
        layers.Dense(8*8*512, use_bias=False),  # 더 큰 기본값으로 시작
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 512)),  # 더 큰 공간 차원에서 시작
        layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

# 판별자 모델 정의
def build_discriminator():
    model = tf.keras.Sequential([
        Input(shape=(64, 64, 3)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# 모델 컴파일 및 훈련
def compile_and_train(generator, discriminator, latent_dim, dataset, epochs):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f'Epoch {epoch+1}/{epochs} completed')

    # 모델 저장 경로 확인 및 생성
    model_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/model'

    # 훈련 후 모델 저장
    generator.save(os.path.join(model_dir, 'generator.keras'))
    discriminator.save(os.path.join(model_dir, 'discriminator.keras'))
    print("Models saved successfully.")


# 데이터셋 및 모델 초기화
BUFFER_SIZE = 200
BATCH_SIZE = 32
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# 훈련 시작
compile_and_train(generator, discriminator, latent_dim, image_dataset, epochs=500)
