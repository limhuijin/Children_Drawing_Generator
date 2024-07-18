import pandas as pd
import os
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 데이터 불러오기
data1 = pd.read_csv('C:/Users/user/Desktop/coding/Painting_Creativity_Tester/csv/Painting_Creativity_Tester_05.csv')
data2 = pd.read_csv('C:/Users/user/Desktop/coding/Painting_Creativity_Tester/csv/Painting_Creativity_Tester_06.csv')
data = pd.concat([data1, data2], ignore_index=True)

# 이미지 파일 경로 설정
image_dir = 'C:/Users/user/Desktop/coding/Painting_Creativity_Tester/images/'
data['image_path'] = [os.path.join(image_dir, f) + '.png' for f in data['FileName']]

# 이미지 로드 및 전처리
class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        img = self.transform(img)
        return img

dataset = ImageDataset(data['image_path'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 생성기(Generator)와 판별기(Discriminator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # [생성기 네트워크 레이어]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [판별기 네트워크 레이어]

# 모델, 손실 함수 및 최적화기 초기화
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 이미지 저장 디렉터리
os.makedirs('generated_images', exist_ok=True)

# 학습 루프
for epoch in range(50):  # 에폭 수
    for i, images in enumerate(dataloader):
        # [학습 루프]

        # 특정 간격으로 생성된 이미지 저장 및 표시
        if (i+1) % 100 == 0:
            with torch.no_grad():
                fake_images = generator(noise)
                save_image(fake_images, f'generated_images/{epoch+1}_{i+1}.png', normalize=True)
    
    # 에폭마다 생성된 이미지 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(fake_images.cpu().data.numpy()[0], (1, 2, 0)))
    plt.axis('off')
    plt.title(f'Images at Epoch {epoch+1}')
    plt.show()

print('Training complete.')
