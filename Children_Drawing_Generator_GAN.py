import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 로드 및 전처리
def load_data():
    dataset_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/dataset/dataset.npy'
    train_images = np.load(dataset_path)
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5  # 정규화 [-1, 1]로
    return train_images

batch_size = 32
epochs = 100000
noise_dim = 100
num_examples_to_generate = 16

# 생성기 모델 정의
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(7*7*256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# 판별기 모델 정의
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Input(shape=[28, 28, 3]),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# 손실 함수 및 옵티마이저 설정
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)

# 모델 체크포인트 디렉토리 설정
checkpoint_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 체크포인트 콜백 설정
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch:02d}.weights.h5")

# 학습 단계 정의
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# 학습 함수 정의
def train(dataset, epochs):
    for epoch in range(epochs):
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(len(dataset)).batch(batch_size)
        gen_loss_avg = 0
        disc_loss_avg = 0
        step_count = 0
        
        for image_batch in train_dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_avg += gen_loss
            disc_loss_avg += disc_loss
            step_count += 1
        
        gen_loss_avg /= step_count
        disc_loss_avg /= step_count

        if (epoch + 1) % 10 == 0:  # 10 에포크마다 저장
            generate_and_save_images(generator, epoch + 1, seed)
            # 모델 체크포인트 저장
            generator.save_weights(checkpoint_prefix.format(epoch=epoch+1))
            discriminator.save_weights(checkpoint_prefix.format(epoch=epoch+1))
        
        print(f'Epoch: {epoch + 1}, gen_loss: {gen_loss_avg}, disc_loss: {disc_loss_avg}')

# 이미지 생성 및 저장 함수 정의
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8))  # numpy() 사용하여 변환
        plt.axis('off')
    os.makedirs('C:/Users/user/Desktop/coding/Children_Drawing_Generator/made_images', exist_ok=True)
    plt.savefig(f'C:/Users/user/Desktop/coding/Children_Drawing_Generator/made_images/image_at_epoch_{epoch:04d}.png')
    plt.close(fig)

# 랜덤 노이즈 생성
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 학습 데이터 로드
train_images = load_data()
print(f'Loaded {len(train_images)} images.')

# 모델 학습
train(train_images, epochs)
