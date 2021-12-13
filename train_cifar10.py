import torch
import torch.nn as nn
import numpy as np
import datetime, os
import torchvision
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from model import *
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
device = 'cuda'
batchsize = 64
maxiter = 1e5
if __name__=='__main__':
    writer = SummaryWriter('./runs/exp1-cifar10/train_{}'.format(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")))
    checkpoint_folder = 'cifar10_checkpoint_res'
    if checkpoint_folder not in os.listdir('.'):
        os.mkdir(os.path.join('./',checkpoint_folder))
#-----------------------------------build model and optimizer-------------------------------------------
    gene32 = Generator(128,blocklist=gene_blocklist, figsize=32, device=device)
    #gene32 = wgan_gp.Generator().to(device)
    dis32 = Discriminator(32, blocklist=block_list, device=device, use_res=True)
    #dis32 = wgan_gp.Discriminator().to(device)
    optim_g = torch.optim.Adam(gene32.parameters(), lr=2e-4)
    optim_d = torch.optim.Adam(dis32.parameters(), lr=2e-4)
#-------------------------------------load and transform dataset------------------------------------------
    transform = T.Compose(
    [T.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, shuffle=True, drop_last=True)
#--------------------------------------train iteration--------------------------------------------------------
    iteration = 0
    writer.add_graph(dis32, gene32(torch.randn(1,128,device=device)))   #show graph of discriminator in tensorboard
    while iteration < maxiter:
        for i_imag, image in enumerate(train_loader):#need reconstruction using iterator
            real_data = image[0].to(device)
            randin = torch.randn(batchsize, 128, device=device)
            gradloss = train_discrminator(dis32, real_data, gene32(randin), optim_d, device=device)
            #train discriminator 5 times each iteration
            if i_imag%5 == 0:
                iteration += 1
                train_generator(gene32, dis32, randin, optim_g)
                #sample generator every 100 iter
                if iteration%100 == 99:
                    with torch.no_grad():
                        print(f'iter:{iteration+1}', end='\r')
                        gene32.eval()
                        sampin = torch.randn(5, 128, device=device)
                        fake_fig = gene32(sampin)*0.5 + 0.5 #transform back to standard rgb
                        writer.add_images('gene', torch.concat([fake_fig, real_data[0,...].unsqueeze(0)*0.5+0.5], dim=0), global_step=iteration)
                        #cal D score on test D_score = mean(D(testdata))'
                        d_score = 0
                        for image in test_loader:
                            gene32.eval()
                            dis32.eval()
                            real_imag = image[0].to(device)
                            d_score -= dis32(real_imag).sum().item()
                    writer.add_scalar('d_score', d_score/len(test_set), iteration)
                if iteration%1000 == 999:
                    torch.save(gene32.state_dict(), os.path.join(os.getcwd(), checkpoint_folder, 'gene_checkpoint_last.pth'))
                    torch.save(dis32.state_dict(), os.path.join(os.getcwd(), checkpoint_folder, 'dis_checkpoint_last.pth'))
                if iteration%(maxiter//10) == (maxiter//10)-1:
                    torch.save(gene32.state_dict(), os.path.join(os.getcwd(), checkpoint_folder, f'gene_checkpoint_{iteration+1}.pth'))
                    torch.save(dis32.state_dict(), os.path.join(os.getcwd(), checkpoint_folder, f'dis_checkpoint_{iteration+1}.pth'))