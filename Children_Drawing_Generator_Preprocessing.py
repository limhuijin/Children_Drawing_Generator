import os
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로 설정
csv_path = 'C:/Users/user/Desktop/coding/Painting_Creativity_Tester/csv/Painting_Creativity_Tester_05.csv'

# 이미지 파일 경로 설정
image_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/'

# CSV 파일 로드
data = pd.read_csv(csv_path)
data['image_path'] = [os.path.join(image_dir, f"{f}.jpg") for f in data['FileName']]  # 여기에 확장자 추가

# 이미지 전처리 함수
def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((28, 28))  # 이미지 크기 조정
        image = np.array(image)
        image = (image - 127.5) / 127.5  # 정규화 [-1, 1]로
        return image
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None

# 이미지를 로드하여 리스트로 저장
image_paths = data['image_path'].tolist()
images = [load_and_preprocess_image(image_path) for image_path in image_paths]
images = [img for img in images if img is not None]  # None 제거
images = np.array(images)

# 데이터셋 저장 경로 설정
dataset_save_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/dataset/dataset.npy'

# 데이터셋 저장
np.save(dataset_save_path, images)
print(f"Dataset saved to {dataset_save_path}")

# TensorFlow 데이터셋으로 변환
batch_size = 2
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)

# 데이터셋 시각화 (첫 10개 이미지 확인)
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, image in enumerate(images[:10]):
    axes[i].imshow((image * 127.5 + 127.5).astype(np.uint8))
    axes[i].axis('off')
plt.show()
