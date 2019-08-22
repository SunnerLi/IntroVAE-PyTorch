from torchvision import datasets, transforms
from model import InferenceModel, Generator
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import os

# Define the parameterize trick
def PARAM(mean, logvar):
    z = mean + torch.exp(logvar).sqrt() * torch.randn_like(logvar)
    return z

# Define the KLD loss
class KLDLoss(torch.nn.Module):
    def forward(self, mu, logvar):
        kl = -0.5 * (1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        kl = kl.sum()
        return kl

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Print(string):
    print("[ IntroVAE ]  {}".format(string))

##########################################################################################################################################
def train(args, G, D):
    # Construct the data loader
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Construct the criterion and optimizer
    crit_AE  = torch.nn.L1Loss(reduction='sum').cuda()
    crit_REG = KLDLoss().cuda()
    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    model_path = os.path.join(args.root_folder, 'Model', 'last.pth')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        G.load_state_dict(state_dict['G'])
        D.load_state_dict(state_dict['D'])
        optim_G.load_state_dict(state_dict['optim_G'])
        optim_D.load_state_dict(state_dict['optim_D'])
        Loss_G_list = state_dict['G_loss']
        Loss_D_list = state_dict['D_loss']
    else:
        Print("Pre-train model is not exist. Train from scratch!")

    # Train!
    fix_z = torch.randn(args.batch_size, args.z_dim).cuda()
    Loss_D_list = []    
    Loss_G_list = []
    for ep in range(args.epochs):
        bar = tqdm(loader)
        loss_D_list = []    
        loss_G_list = []
        if ep != 0:
            for it, (x, _) in enumerate(bar):
                # forward for inference model revision
                x = x.cuda()
                z_mu, z_logvar = D(x)
                z_p = torch.randn(x.size(0), args.z_dim).cuda()
                x_rec = G(PARAM(z_mu, z_logvar))
                x_p = G(z_p)
                z_mu_r, z_logvar_r = D(x_rec.detach())
                z_mu_pp, z_logvar_pp = D(x_p.detach())

                # backward inference model
                optim_D.zero_grad()
                L_adv_E = crit_REG(z_mu, z_logvar) + args.alpha * (
                    torch.clamp(args.m - crit_REG(z_mu_r, z_logvar_r), min=0) + \
                    torch.clamp(args.m - crit_REG(z_mu_pp, z_logvar_pp), min=0)
                )
                L_AE = crit_AE(x_rec, x) * args.beta
                loss_D = L_adv_E + L_AE
                loss_D_list.append(loss_D.item())
                loss_D.backward()
                optim_D.step()

                # forward for generator revision
                z_mu, z_logvar = D(x)
                z_p = torch.randn(x.size(0), args.z_dim).cuda()
                x_rec = G(PARAM(z_mu, z_logvar))
                x_p = G(z_p)
                z_mu_r, z_logvar_r = D(x_rec)
                z_mu_pp, z_logvar_pp = D(x_p)

                # backward generator
                optim_G.zero_grad()
                L_adv_G = args.alpha * (crit_REG(z_mu_r, z_logvar_r) + crit_REG(z_mu_pp, z_logvar_pp))
                L_AE = crit_AE(x_rec, x) * args.beta
                loss_G = L_adv_G + L_AE
                loss_G_list.append(loss_G.item())
                loss_G.backward()
                optim_G.step()

                # Show the information
                if it == len(bar)-1:
                    bar.set_description("Avg loss_D: {:.6f}   Avg loss_G: {:.6f}".format(np.mean(loss_D_list), np.mean(loss_G_list)))
                else:
                    bar.set_description("loss_D: {:.6f}   loss_G: {:.6f}".format(loss_D_list[-1], loss_G_list[-1]))
            Loss_D_list.append(np.mean(loss_D_list))
            Loss_G_list.append(np.mean(loss_G_list))
        # Save the result
        inference(args, G, z = fix_z, n = args.batch_size, file_name=os.path.join(args.root_folder, 'Sample', '%03d.png'%ep))
        state_dict = {
            'G': G.state_dict(), 
            'D': D.state_dict(), 
            'G_loss': Loss_G_list, 
            'D_loss': Loss_D_list,
            'optim_G': optim_G.state_dict(),
            'optim_D': optim_D.state_dict()
        }
        torch.save(state_dict, f=os.path.join(args.root_folder, 'Model', 'last.pth'))

    # Plot the loss curve
    plt.figure()
    plt.plot(range(len(Loss_G_list)), Loss_G_list)
    plt.savefig(os.path.join(args.root_folder, 'loss_G.png'))
    plt.clf()
    plt.figure()
    plt.plot(range(len(Loss_D_list)), Loss_D_list)
    plt.savefig(os.path.join(args.root_folder, 'loss_D.png'))


def inference(args, G, z = None, n = 64, file_name = '???'):
    if z is None:
        z = torch.randn(n, args.z_dim).cuda()
    G.eval()
    with torch.no_grad():
        fake_img = G(z)
        save_image(fake_img, file_name, normalize=True)
    G.train()

##########################################################################################################################################
if __name__ == '__main__':
    # Parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha'       , type = float, default = 0.25)
    parser.add_argument('--beta'        , type = float, default = 0.05)
    parser.add_argument('--m'           , type = float, default = 120.0)
    parser.add_argument('--z_dim'       , type = int, default = 128)
    parser.add_argument('--batch_size'  , type = int, default = 16)
    parser.add_argument('--epochs'      , type = int, default = 1)
    parser.add_argument('--n'           , type = int, default = -1, help = 'The number of image you want to sample. -1 for only training. Otherwise for only inference')
    parser.add_argument('--root_folder' , type = str, default = 'train_result')
    args = parser.parse_args()
    Print('------------------------------------------')
    for key, val in vars(args).items():
        Print("{:>20} : {}".format(key, val))
    Print('------------------------------------------')

    # Create the folder
    mkdir(args.root_folder)
    if args.n == -1:
        mkdir(os.path.join(args.root_folder, 'Model'))
        mkdir(os.path.join(args.root_folder, 'Sample'))
    else:
        mkdir(os.path.join(args.root_folder, 'Inference'))

    # Work!
    G = Generator(z_dim=args.z_dim, out_channels=1).cuda()
    D = InferenceModel(in_channels=1, z_dim=args.z_dim).cuda()
    if args.n == -1:
        train(args, G, D)
    else:
        model_path = os.path.join(args.root_folder, 'Model', 'last.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            G.load_state_dict(state_dict['G'])
            inference(args, G, n = args.n, file_name=os.path.join(args.root_folder, 'Inference', 'sample.png'))
        else:
            raise Exception("The pre-trained model is not exist...")