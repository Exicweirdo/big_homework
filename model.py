import torch
import torch.nn as nn
import numpy as np
#def resblock
class resblock1(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size, device='cpu', stride=1) -> None:
        super(resblock1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=False, device=device)
        self.batchnorm = nn.BatchNorm2d(out_channels, device=device)
        self.identity = nn.Conv2d(in_channels, out_channels, 1, stride, padding='same', bias=False, device=device)
    def forward(self, x):
        hid = self.conv2d(x)
        hid = self.batchnorm(hid)
        out = self.identity(x) + nn.ReLU()(hid)
        return out
# generatar model
res_list = [
    {'kernel':([3,3],2), 'resamp':'up', 'out_channel':512},
    {'kernel':([3,3],2), 'resamp':'up', 'out_channel':256},
    {'kernel':([3,3],2), 'resamp':'up', 'out_channel':128},
    {'kernel':([3,3],2), 'resamp':'up', 'out_channel':64},
]
block_list = [
    {'kernel':([3,3],2), 'resamp':'down', 'out_channel':128},
    {'kernel':([3,3],2), 'resamp':'down', 'out_channel':256},
    {'kernel':([3,3],2), 'resamp':'down', 'out_channel':512},
    {'kernel':([3,3],2), 'resamp':'down', 'out_channel':512},
]
class Generator(torch.nn.Module):
    def __init__(self, input_shape:int, reslist, device='cpu') -> None:
        super(Generator, self).__init__()
        self.device = device
        self.preprocess = nn.Sequential(
            nn.Linear(input_shape, 4*4, device=device),
            nn.ReLU()
        )
        in_channel = 1
        Layers = []
        for layer in reslist:
            #change channels
            Layers.append(nn.Conv2d(in_channel, layer['out_channel'], 1, padding='same', bias=False, device=device))
            Layers.append(nn.BatchNorm2d(layer['out_channel'],device=device))
            in_channel = layer['out_channel']
            #apply resblock
            for i in range(layer['kernel'][1]):
                Layers.append(resblock1(in_channel, in_channel, layer['kernel'][0], device=device))
            #resampling
            if layer['resamp']=='up':
                Layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif layer['resamp']=='down':
                Layers.append(nn.AvgPool2d(kernel_size=2))
        #give rgb data
        Layers.extend([
            nn.Conv2d(in_channel, 3, 1, padding='same', bias=False,device=device),
            nn.Tanh()
        ])
        self.Seq = nn.Sequential(*Layers)
        self.init_parameters()
    def forward(self, x) -> torch.Tensor:
        res_in = self.preprocess(x).view(-1, 1, 4, 4)
        out = self.Seq(res_in)
        return out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
        return 0

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, blocklist, device = 'cpu') -> None:
        super(Discriminator, self).__init__()
        self.device = device
        self.preprocess = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same', device=device)
        in_channel = 64
        Layers = []
        for layer in blocklist:
            #change channels
            Layers.append(nn.Conv2d(in_channel, layer['out_channel'], 1, padding='same', bias=False, device=device))
            Layers.append(nn.BatchNorm2d(layer['out_channel'],device=device))
            in_channel = layer['out_channel']
            #apply resblock
            for i in range(layer['kernel'][1]):
                Layers.append(resblock1(in_channel, in_channel, layer['kernel'][0], device=device))
            #resampling
            if layer['resamp']=='up':
                Layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif layer['resamp']=='down':
                Layers.append(nn.AvgPool2d(kernel_size=2))
        self.seq = nn.Sequential(*Layers)
        self.outlinear = nn.Linear(in_features=4*4*in_channel, out_features=1,device=device)
        self.outchannel = in_channel
        self.init_parameters()

    def forward(self, x) -> torch.Tensor:
        res_in = self.preprocess(x)
        res_out = self.seq(res_in)
        out = self.outlinear(res_out.view(-1, 4*4*self.outchannel))
        return(out)
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
        return 0


def get_param_num(model:nn.Module):
    param_num = sum(p.numel() for p in model.parameters())
    trainable_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return{'Total':param_num, 'Trainable':trainable_param_num}

def train_discrminator(model:Discriminator, real_data:torch.Tensor, fake_data:torch.Tensor, optimizer:torch.optim.Optimizer, k = 2, p = 6):
    model.train()
    batch_size = real_data.size(0)
    score_loss = torch.sum(model(real_data)-model(fake_data))/batch_size
    #compute gradient loss
    mixconst = torch.rand(real_data.size(0))
    x_mix = torch.tensordot(torch.diag(mixconst), real_data, dims=[[0],[0]]) \
        + torch.tensordot(torch.diag(1-mixconst), fake_data, dims=[[0],[0]])
    x_mix.requires_grad_()
    x_mix.retain_grad()
    model.zero_grad()
    grad_mix = torch.autograd.grad(model(x_mix), x_mix, create_graph=True) 
    gradient_loss = k*torch.norm(grad_mix)**p
    #add loss
    loss = score_loss + gradient_loss
    #train
    loss.backward()
    optimizer.step()

def train_generator(g_model:Generator, d_model:Discriminator, randinput:torch.Tensor, optimizer:torch.optim.Optimizer):
    g_model.train()
    d_model.eval()
    g_model.zero_grad()
    batchsize = randinput.size(0)
    loss = torch.sum(d_model(g_model(randinput)))/batchsize
    loss.backward()
    optimizer.step()

if __name__=='__main__':
    gene64 = Generator(128, reslist=res_list)
    dis64 = Discriminator(64, blocklist=block_list)
    print(dis64)
    fake = gene64(torch.randn(1,128))
    print(fake.size())
    print(fake.requires_grad_())
    fake.retain_grad()
    score_fake = dis64(fake)
    grad = torch.autograd.grad(score_fake, fake, create_graph=True)[0]
    print(grad.size())
    print(get_param_num(gene64))
    print(get_param_num(dis64))