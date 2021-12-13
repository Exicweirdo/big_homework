import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
#structure of generator and discriminator
gene_blocklist = [
    {'kernel':2, 'out_channel':256, 'stride':2},
    {'kernel':2, 'out_channel':128, 'stride':2},
    {'kernel':2, 'out_channel':64, 'stride':2},
    {'kernel':2, 'out_channel':32, 'stride':2},
]
block_list = [
    {'kernel':3, 'out_channel':128, 'stride':2},
    {'kernel':3, 'out_channel':256, 'stride':2},
    {'kernel':3, 'out_channel':512, 'stride':2},
]
#def convtrans resblock
class Tresblock1(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size, device='cpu', stride=1) -> None:
        super(Tresblock1, self).__init__()
        self.convT2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
        padding=(kernel_size-stride)//2, device=device)
        self.batchnorm = nn.BatchNorm2d(out_channels, device=device)
        self.identity = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False, device=device)
        )
    def forward(self, x):
        hid = self.convT2d(x)
        hid = self.batchnorm(hid)
        out = nn.ReLU()(hid)  + self.identity(x)
        return out
#resblock without bn
class resblock(torch.nn.Module):
    def __init__(self, in_channel:int, out_channel:int, stride:int = 1, dilation:int=1, device = 'cpu') -> None:
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False, device=device)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, bias=False, device=device)
        self.downsample = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False, device=device)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out+= self.downsample(x)
        out = self.relu(out)
        return out

# generatar model
class Generator(torch.nn.Module):
    def __init__(self, input_shape:int, blocklist, figsize:int, device='cpu') -> None:
        super(Generator, self).__init__()
        self.device = device
        in_channel = blocklist[0]['out_channel']*2
        self.input_channel = in_channel
        Layers = []
        self.input_size = figsize
        for layer in blocklist:
            #change channels
            #apply resblock
            Layers.append(Tresblock1(in_channel, layer['out_channel'], layer['kernel'], stride=2, device=device))
            self.input_size = self.input_size//layer['stride']
            in_channel = layer['out_channel']
        #give rgb data
        Layers.extend([
            nn.ConvTranspose2d(in_channel, 3, kernel_size=2, stride=2, device=device),
            nn.Tanh()
        ])
        self.input_size = self.input_size//2
        self.preprocess = nn.Sequential(
            nn.Linear(input_shape, blocklist[0]['out_channel']*2*self.input_size**2, device=device),
            nn.ReLU()
        )
        self.Seq = nn.Sequential(*Layers)
        self.init_parameters()
    def forward(self, x) -> torch.Tensor:
        res_in = self.preprocess(x).view(-1, self.input_channel, self.input_size, self.input_size)
        out = self.Seq(res_in)
        return out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return 0

class Discriminator(torch.nn.Module):
    def __init__(self, input_size:int, blocklist, device = 'cpu', use_res = False) -> None:
        super(Discriminator, self).__init__()
        self.device = device
        #self.preprocess = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same', device=device)
        in_channel = 3
        self.figsize = input_size
        Layers = []
        if use_res==False:
            for layer in blocklist:
                #change channels
                Layers.extend([
                    nn.Conv2d(in_channel, layer['out_channel'], layer['kernel'],stride=layer['stride'], padding=1, device=device),
                    nn.LeakyReLU()
                ])
                #resampling
                in_channel = layer['out_channel']
                self.figsize = self.figsize//layer['stride']
        else:
            for layer in block_list:
                Layers.append(resblock(in_channel, layer['out_channel'], stride=layer['stride'], device=device))
                in_channel = layer['out_channel']
                self.figsize = self.figsize//layer['stride']
             
        self.seq = nn.Sequential(*Layers)
        self.outlinear = nn.Linear(in_features=self.figsize**2*in_channel, out_features=1,device=device)
        self.outchannel = in_channel
        self.init_parameters()

    def forward(self, x) -> torch.Tensor:
        #res_in = self.preprocess(x)
        res_out = self.seq(x)
        out = self.outlinear(res_out.view(-1, self.figsize**2*self.outchannel))
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

def train_discrminator(model:Discriminator, real_data:torch.Tensor, fake_data:torch.Tensor, optimizer:torch.optim.Optimizer, k = 2, p = 6, device = 'cpu'):
    model.train()
    score_loss = model(fake_data).mean()-model(real_data).mean()#torch.sum(model(fake_data)-model(real_data))/batch_size
    #compute gradient loss
    mixconst = torch.rand(real_data.size(0), device=device)
    x_mix = torch.tensordot(torch.diag(mixconst), real_data, dims=[[0],[0]]) \
        + torch.tensordot(torch.diag(1-mixconst), fake_data, dims=[[0],[0]])
    x_mix.requires_grad_()
    model.zero_grad()
    model.eval()
    grad_mix = torch.autograd.grad(model(x_mix).sum(), x_mix, create_graph=True, retain_graph=True)[0]
    model.train()
    #grad_mix_check =  torch.autograd.grad(model(x_mix), x_mix, grad_outputs=torch.ones(x_mix.size()).to(device), create_graph=True, retain_graph=True)[0]
    gradient_loss = (k*torch.sum(grad_mix**2, dim=[1,2,3])**(p/2)).mean()
    #add loss
    loss = score_loss + gradient_loss
    #train
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return gradient_loss
def train_generator(g_model:Generator, d_model:Discriminator, randinput:torch.Tensor, optimizer:torch.optim.Optimizer):
    g_model.train()
    d_model.eval()
    g_model.zero_grad()
    loss = -d_model(g_model(randinput)).mean()#torch.sum(-d_model(g_model(randinput)))/batchsize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__=='__main__':
    device = 'cuda'
    writer = SummaryWriter('./runs/exp1-cifar10/no_batchnorm')
    batchsize = 64
    maxiter = 1e5
#--------------------------------------------------------------------------------------
    gene32 = Generator(128,blocklist=gene_blocklist, figsize=32, device=device)
    #gene32 = wgan_gp.Generator().to(device)
    dis32 = Discriminator(32, blocklist=block_list, device=device)
    #dis32 = wgan_gp.Discriminator().to(device)
    optim_g = torch.optim.Adam(gene32.parameters(), lr=2e-4)
    optim_d = torch.optim.Adam(dis32.parameters(), lr=2e-4)
#-------------------------------------dataset------------------------------------------
    transform = T.Compose(
    [T.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, shuffle=True, drop_last=True)
    
    iteration = 0
    writer.add_graph(dis32, gene32(torch.randn(1,128,device=device)))
    while iteration < maxiter:
        for i_imag, image in enumerate(train_loader):
            real_data = image[0].to(device)
            randin = torch.randn(batchsize, 128, device=device)
            gradloss = train_discrminator(dis32, real_data, gene32(randin), optim_d, device=device)
            if i_imag%5 == 0:
                iteration += 1
                train_generator(gene32, dis32, randin, optim_g)
                #estimate d-score
                if iteration%100 == 99:
                    with torch.no_grad():
                        print(f'iter:{iteration+1}', end='\r')
                        gene32.eval()
                        sampin = torch.randn(5, 128, device=device)
                        fake_fig = gene32(sampin)*0.5 + 0.5
                        print(fake_fig.max().item(), fake_fig.min().item())
                        writer.add_images('gene', torch.concat([fake_fig, real_data[0,...].unsqueeze(0)*0.5+0.5], dim=0), global_step=iteration)
                        d_score = 0
                        for image in test_loader:
                            gene32.eval()
                            dis32.eval()
                            real_imag = image[0].to(device)
                            d_score -= dis32(real_imag).sum().item()
                    writer.add_scalar('d_cost', d_score/len(test_set), iteration)
                if iteration%1000 == 999:
                    torch.save(gene32.state_dict(), 'gene_checkpoint_last.pth')
                    torch.save(dis32.state_dict(), 'dis_checkpoint_last.pth')
                if iteration%(maxiter//10) == (maxiter//10)-1:
                    torch.save(gene32.state_dict(), f'gene_checkpoint_{iteration+1}.pth')
                    torch.save(dis32.state_dict(), f'dis_checkpoint_{iteration+1}.pth')
            '''
            if i_imag%10==0:
                gene32.eval()
                dis32.eval()
                sampin = torch.randn(5, 128, device=device)
                d_loss = (-dis32(real_data)+dis32(gene32(randin))).mean().item()
                g_loss = -dis32(gene32(sampin)).mean().item()
                print('epoch {}:{:.2%}  loss_d:{:.5f}'.format(i, i_imag*batchsize/50000, d_loss))
                print('grad:{}'.format(gradloss))
                writer.add_scalar('GLoss',g_loss, iteration)
                writer.add_scalar('DLoss',d_loss, iteration)
            '''
            #estimate d-score