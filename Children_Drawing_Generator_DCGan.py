import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 이미지 데이터의 올바른 경로로 수정
path = "C:/Users/user/Desktop/coding/Children_Drawing_Generator/images"
if not os.path.exists(path):
    raise FileNotFoundError(f"Couldn't find any class folder in {path}")

image_size = 128  # 더 높은 해상도로 변경
batch_size = 32
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(path, transform=tt.Compose([
    tt.Resize(image_size),
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats),
    tt.RandomHorizontalFlip(p=0.5)
]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def save_images(images, file_path, nmax=64):
    save_image(denorm(images.detach()[:nmax]), file_path, nrow=8)
    print(f'Saved images to {file_path}')

def get_default_device():
    return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

device = get_default_device()
print(f'Using device: {device}')

train_dl = DeviceDataLoader(train_dl, device)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
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

latent_size = 100
discriminator = Discriminator().to(device)
generator = Generator(latent_size).to(device)

# Create optimizers
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.FloatTensor(real_preds.size()).fill_(0.9).to(device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    fake_targets = torch.FloatTensor(fake_preds.size()).fill_(0.1).to(device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt):
    opt.zero_grad()
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    preds = discriminator(fake_images)
    targets = torch.ones(preds.size(), device=device)
    loss = F.binary_cross_entropy(preds, targets)
    loss.backward()
    opt.step()
    return loss.item()

sample_dir = 'C:/Users/user/Desktop/coding/Children_Drawing_Generator/generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = '{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

save_samples(0, fixed_latent, show=False)

def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, losses_g, losses_d, real_scores, fake_scores):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
        'losses_g': losses_g,
        'losses_d': losses_d,
        'real_scores': real_scores,
        'fake_scores': fake_scores
    }
    torch.save(checkpoint, os.path.join(sample_dir, f'checkpoint_{epoch}.pth'))
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(generator, discriminator, opt_g, opt_d, epoch):
    checkpoint = torch.load(os.path.join(sample_dir, f'checkpoint_{epoch}.pth'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    losses_g = checkpoint['losses_g']
    losses_d = checkpoint['losses_d']
    real_scores = checkpoint['real_scores']
    fake_scores = checkpoint['fake_scores']
    print(f"Checkpoint loaded from epoch {epoch}")
    return losses_g, losses_d, real_scores, fake_scores

def fit(epochs, lr, start_epoch=1, save_interval=10):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    for epoch in range(start_epoch, epochs + start_epoch):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch, epochs + start_epoch - 1, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images (매 에포크마다 저장)
        save_samples(epoch, fixed_latent, show=False)

        # Save checkpoint every 'save_interval' epochs (10 에포크마다 저장)
        if epoch % save_interval == 0:
            save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, losses_g, losses_d, real_scores, fake_scores)
    
    return losses_g, losses_d, real_scores, fake_scores

if __name__ == '__main__':
    lr = 0.0002
    epochs = 5000

    # 체크포인트가 존재하는 경우 이를 불러오고 학습을 이어갑니다.
    latest_checkpoint = max([int(f.split('_')[1].split('.')[0]) for f in os.listdir(sample_dir) if 'checkpoint_' in f], default=0)
    if latest_checkpoint > 0:
        print(f"Resuming training from epoch {latest_checkpoint}")
        losses_g, losses_d, real_scores, fake_scores = load_checkpoint(generator, discriminator, opt_g, opt_d, latest_checkpoint)
        history = fit(epochs, lr, start_epoch=latest_checkpoint + 1)
    else:
        print("Starting training from scratch")
        history = fit(epochs, lr)

    losses_g, losses_d, real_scores, fake_scores = history
