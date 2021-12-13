import matplotlib.pyplot as plt
from torch import torch_version
from matplotlib import animation
import torchvision
import torch
from model import *

gene_blocklist = [
    {'kernel':2, 'out_channel':256, 'stride':2},
    {'kernel':2, 'out_channel':128, 'stride':2},
    {'kernel':2, 'out_channel':64, 'stride':2},
    {'kernel':2, 'out_channel':32, 'stride':2},
]
device = 'cuda'
batchsize = 256
if __name__ == '__main__':
    #build model
    gene32 = Generator(128,blocklist=gene_blocklist, figsize=32, device=device)
    #get randinput
    randinput = torch.randn(batchsize, 128, device=device)
    gene32.load_state_dict(torch.load('./cifar10_checkpoint/gene_checkpoint_100000.pth'))
    gene32.eval()
    with torch.no_grad():
        figtensor = gene32(randinput)*0.5+0.5   #batchsize*3*32*32 output
        grid = torchvision.utils.make_grid(figtensor, padding=0, nrow=16)
        torchvision.utils.save_image(grid, './grid.png')
        tensorlist = []
        for i in range(10):
            gene32.load_state_dict(torch.load(f'./cifar10_checkpoint/gene_checkpoint_{i+1}0000.pth'))
            figtensor = gene32(randinput)*0.5+0.5   #batchsize*3*32*32 output
            grid = torchvision.utils.make_grid(figtensor, padding=0, nrow=16).cpu().detach().numpy()
            tensorlist.append(np.transpose(grid, (1,2,0)))
    
    #render a gif
    fig, ax = plt.subplots()
    def ani(i):
        ax.cla()
        plt.axis('off')
        ax.imshow(tensorlist[i], interpolation='none')
        
    gif = animation.FuncAnimation(fig=fig, func=ani, frames=10, interval = 200)
    gif.save('ani.gif', writer='pillow')
    