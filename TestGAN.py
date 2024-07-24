import os
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# 경로 설정
checkpoint_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/generated/checkpoint_400.pth'  # 체크포인트 파일 경로
output_image_base_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/generated_image'  # 생성된 이미지 저장 경로 (확장자 제외)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 정의 (디스크리미네이터와 제너레이터)
latent_size = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 모델 인스턴스화 및 디바이스 이동
generator = Generator().to(device)

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])

# 넘버링된 파일 이름 생성 함수
def get_new_file_path(base_path, ext='.png'):
    counter = 0
    while True:
        file_path = f"{base_path}_{counter}{ext}"
        if not os.path.exists(file_path):
            return file_path
        counter += 1

# 이미지 생성 함수
def generate_and_save_images(generator, latent_size, output_image_base_path, num_images=100, show=True):
    images = []
    for i in range(num_images): 
        # 새로운 랜덤 노이즈 생성
        latent_tensors = torch.randn(1, latent_size, 1, 1, device=device)
        # 이미지 생성
        with torch.no_grad():
            fake_image = generator(latent_tensors)
        images.append(fake_image)
        # 넘버링된 파일 경로 생성
        output_image_path = get_new_file_path(output_image_base_path)
        # 이미지 저장
        save_image(fake_image, output_image_path)
        print(f'Saved generated image to {output_image_path}')
    
    # 이미지 시각화
    if show:
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(make_grid(torch.cat(images), nrow=5).permute(1, 2, 0) * 0.5 + 0.5)  # denormalize
        plt.show()

# 이미지 생성 및 저장
generate_and_save_images(generator, latent_size, output_image_base_path=output_image_base_path, num_images=100)
