# 라이브러리 임포트
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import os

# CSV 파일 불러오기
csv_file1 = '/mnt/data/Painting_Creativity_Tester_05.csv'
csv_file2 = '/mnt/data/Painting_Creativity_Tester_06.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

import ace_tools as tools; tools.display_dataframe_to_user("Painting Creativity Tester 05", df1)
tools.display_dataframe_to_user("Painting Creativity Tester 06", df2)

# 이미지 데이터 경로 설정
image_directory = '/path/to/childrens_drawings'

# 이미지 데이터 전처리를 위한 ImageDataGenerator 설정
datagen = ImageDataGenerator(
    rescale=1.0/255.0, # 픽셀 값 정규화
    validation_split=0.2 # 훈련 및 검증 세트로 분할
)

# 훈련 데이터 로드
train_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128), # 이미지를 128x128로 리사이즈
    batch_size=32,
    class_mode='input', # 오토인코더 사용
    subset='training'
)

# 검증 데이터 로드
validation_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    subset='validation'
)

# 오토인코더 모델 정의
input_img = Input(shape=(128, 128, 3))

# 인코더
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# 디코더
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# 오토인코더 모델 컴파일
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# 모델 훈련
history = autoencoder.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# 새로운 이미지 생성
num_images_to_generate = 5
random_latent_vectors = np.random.normal(size=(num_images_to_generate, 128, 128, 3))

generated_images = autoencoder.predict(random_latent_vectors)

# 생성된 이미지를 저장할 디렉토리 설정
save_directory = '/path/to/save/generated_images'
os.makedirs(save_directory, exist_ok=True)

# 생성된 이미지 저장
for i, img in enumerate(generated_images):
    save_path = os.path.join(save_directory, f'generated_image_{i}.png')
    tf.keras.preprocessing.image.save_img(save_path, img)
