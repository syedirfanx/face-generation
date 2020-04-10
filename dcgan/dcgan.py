from google.colab import files
src = list(files.upload().values())[0]
open('mylib.py','wb').write(src)
import mylib

# Commented out IPython magic to ensure Python compatibility.
data_dir = 'image/Face'

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests

# %matplotlib inline

import torch
from torchvision import datasets
from torchvision import transforms
from google.colab import drive
drive.mount('/content/gdrive')

def get_dataloader(batch_size, image_size, data_dir='/content/gdrive/My Drive/image/'):

    transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])
    data = datasets.ImageFolder(data_dir,transform = transform)
    
    dataLoader = torch.utils.data.DataLoader(data,batch_size = batch_size, shuffle= True)
    
    return dataLoader

batch_size = 64
img_size = 32

celeba_train_loader = get_dataloader(batch_size, img_size)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(celeba_train_loader)
images, _ = dataiter.next()
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])

def scale(x, feature_range=(-1, 1)):
    x = x*2 - 1
    return x

img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())

import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels,out_channels,kernel_size=4,stride = 2,padding=1,batch_norm = True):
  
    layer = list()
    layer.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding))
  
    if(batch_norm):
        layer.append(nn.BatchNorm2d(out_channels))
  
    return nn.Sequential(*layer)

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        self.conv_dim  = conv_dim
        
        super(Discriminator, self).__init__()

        self.conv1 = conv(3,conv_dim,4,2)     #16x16x32
        self.conv2 = conv(conv_dim,conv_dim*2,4,2)    #8x8x64
        self.conv3 = conv(conv_dim*2,conv_dim*4,4,2)   #4x4x128
        self.conv4 = conv(conv_dim*4,conv_dim*8,4,2,batch_norm=False)  #2x2x256 
        
        self.fc1 = nn.Linear(2*2*conv_dim*8,512)
        self.fc2 = nn.Linear(512,1)
        
        self.drp = nn.Dropout(0.3)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        
        x = x.reshape(-1,2*2*self.conv_dim*8)
        x = F.leaky_relu(self.fc1(x))
        x = self.drp(x)
        x = self.fc2(x)
        
        return x

tests.test_discriminator(Discriminator)

def deconv(in_channels,out_channels,kernel_size=4,stride = 2,padding=1,batch_norm = True):
  
    layer = list()
    layer.append(nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding))

    if(batch_norm):
        layer.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layer)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):

        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim = 1024
        self.fc1 = nn.Linear(z_size,1*1*conv_dim)
        
        self.conv1 = deconv(conv_dim,conv_dim//2,4,2)     #2*2*conv_dim/2
        self.conv2 = deconv(conv_dim//2,conv_dim//4)            #4*4*conv_dim/4
        self.conv3 = deconv(conv_dim//4,conv_dim//8)            #8*8*conv_dim/8
        self.conv4 = deconv(conv_dim//8,conv_dim//16)            #16*16*3
        self.conv5 = deconv(conv_dim//16,3,batch_norm = False)   #32*32*3

    def forward(self, x):
        
        x = F.leaky_relu(self.fc1(x))
        x = x.reshape(-1,self.conv_dim,1,1)
        
        x = F.leaky_relu(self.conv1(x))
        
        x = F.leaky_relu(self.conv2(x))
        
        x = F.leaky_relu(self.conv3(x))
        
        x = F.leaky_relu(self.conv4(x))
        
        x = torch.tanh(self.conv5(x))
        
        
        return x

tests.test_generator(Generator)

def weights_init_normal(m):

    classname = m.__class__.__name__
    
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G

d_conv_dim = 64
g_conv_dim = 1024
z_size = 100 

D, G = build_network(d_conv_dim, g_conv_dim, z_size)

import torch

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')

def real_loss(D_out):

    criterian = nn.BCEWithLogitsLoss()
    target = torch.ones(*D_out.shape)
    
    if train_on_gpu:
        target = target.cuda()
    loss = criterian(D_out,target)
    return loss

def fake_loss(D_out):

    criterian = nn.BCEWithLogitsLoss()
    target = torch.zeros(*D_out.shape)
    
    if train_on_gpu:
        target = target.cuda()
    loss = criterian(D_out,target)
    return loss

import torch.optim as optim

d_optimizer = optim.Adam(D.parameters(),lr = 0.0002)
g_optimizer = optim.Adam(G.parameters(),lr = 0.0002)

def train(D, G, n_epochs, print_every=50):

    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    for epoch in range(n_epochs):

        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            d_optimizer.zero_grad()
            
            if train_on_gpu:
                real_images = real_images.cuda()
                
            d_out = D(real_images)
            d_real_loss = real_loss(d_out)
            
            z = np.random.uniform(-1, 1, size=(sample_size, z_size))
            z = torch.from_numpy(z).float()
            
            if train_on_gpu:
                z = z.cuda()
            fake_out = G(z)
            
            g_fake_loss = fake_loss(D(fake_out))
            
            d_loss = d_real_loss + g_fake_loss 
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            
            g_out = G(z)
            d_real_out = D(g_out)
            g_loss = real_loss(d_real_out)
            
            g_loss.backward()
            g_optimizer.step()

            if batch_i % print_every == 0:
                losses.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))

        G.eval()
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()

    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses

n_epochs = 10

losses = train(D, G, n_epochs=n_epochs)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

_ = view_samples(-1, samples)

