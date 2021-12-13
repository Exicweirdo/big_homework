import torchvision
import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
a = torchvision.datasets.CelebA('./data', download=True)