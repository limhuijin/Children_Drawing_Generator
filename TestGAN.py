import os
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# 경로 설정
sample_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/generated'
checkpoint_path = os.path.join(sample_dir, 'checkpoint_550.pth')  # 체크포인트 파일 경로 설정
output_image_path = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/generated_image_01.png'  # 생성된 이미지 저장 경로

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 정의 (디스크리미네이터와 제너레이터)
latent_size = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
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
        return self.net(x)

# 모델 인스턴스화 및 디바이스 이동
generator = Generator().to(device)

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path)
generator.load_state_dict(checkpoint['generator_state_dict'])

# 이미지 생성 함수
def generate_and_save_images(generator, latent_size, num_images, output_image_path, show=True):
    # 랜덤 노이즈 생성
    latent_tensors = torch.randn(num_images, latent_size, 1, 1, device=device)
    # 이미지 생성
    with torch.no_grad():
        fake_images = generator(latent_tensors)
    # 이미지 저장
    save_image(fake_images, output_image_path, nrow=8)
    print(f'Saved generated images to {output_image_path}')
    # 이미지 시각화
    if show:
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()

# 이미지 생성 및 저장
generate_and_save_images(generator, latent_size, num_images=64, output_image_path=output_image_path)
