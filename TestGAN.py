import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 저장된 생성자 모델을 불러오기
generator = tf.keras.models.load_model('C:/Users/user/Desktop/coding/Children_Drawing_Generator/model/GAN/generator.keras')

# 잠재 공간 차원 설정 (모델을 훈련할 때 사용된 차원)
latent_dim = 100
num_images = 10

# 이미지 생성
def generate_images(model, num_images, latent_dim):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = model(noise, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(num_images):
        plt.subplot(5, 2, i + 1)
        # 이미지를 [0, 1] 범위로 조정
        image = (generated_images[i, :, :, :] + 1) / 2
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 생성된 이미지 표시
generate_images(generator, num_images, latent_dim)
