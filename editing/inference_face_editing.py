import argparse
import os
from tqdm import tqdm
# import numpy as np
import torch
from PIL import Image
# from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
import random


import sys
sys.path.append(".")
sys.path.append("..")

# from configs import data_configs
# from datasets.inference_dataset import InferenceDataset
# from editing.face_editor import FaceEditor
# from options.test_options import TestOptions
# from utils.inference_utils import run_inversion
from models.DynaGAN_nada import SG2Generator

edit_paths = {
        "age": "/kuacc/users/aanees20/hpc_run/DynaGAN/editing/interfacegan_directions/age.pt",
        "smile": "/kuacc/users/aanees20/hpc_run/DynaGAN/editing/interfacegan_directions/smile.pt",
        "pose": "/kuacc/users/aanees20/hpc_run/DynaGAN/editing/interfacegan_directions/pose.pt"
    }

edit_latents = {}
edit_latents["unchanged"] = torch.zeros((1,18,512)).float().cuda()
for edit in edit_paths:
    edit_latents[edit] = torch.load(edit_paths[edit]).cuda()
    if edit_latents[edit].ndim == 2:
        edit_latents[edit] = edit_latents[edit].unsqueeze(1).repeat(1,18,1)
edit_latents = torch.cat(list(edit_latents.values()))

def sample_latents(generator, latent_avg, batch_size=1, n_latents=18, truncation=0.5):
    sample_z = torch.randn(batch_size, 512, device="cuda")
    w_styles = generator.style([sample_z])[0]
    output_latents = truncation * (w_styles - latent_avg) + latent_avg
    output_latents = output_latents.unsqueeze(1).repeat(1, n_latents, 1)
    return output_latents

def get_embed_from_img_name(target_path, img_name, clip_preprocess, clip_model, batch_size=1):
    # batch_size = 1
    img1 = os.path.join(target_path, img_name)
    img1 = f"{img1}/{img_name}.jpg"
    img_cond_1 = Image.open(img1)
    img_preprocess_1 = clip_preprocess(img_cond_1).unsqueeze(0).cuda()
    with torch.no_grad():
        img_embed_1 = clip_model.encode_image(img_preprocess_1).to(torch.float32)
    img_embed_1 = img_embed_1.repeat(batch_size,1) ###
    return img_embed_1, img_cond_1

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def run(args):
    # test_opts = TestOptions().parse()
    root_path = "/scratch/users/aanees20/hpc_run/DynaGAN/target_data"
    target = "nada_data_folder_wise/"
    target_path = os.path.join(root_path, target)
    
    exp_dir = args.out_path #"/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/latent_editing/"
    checkpoint_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_4/checkpoint/011500.pt"
    device = "cuda"
    batch_size = 1

    out_path_results = os.path.join(exp_dir, f'editing_results_all_{args.factor}')
    # out_path_coupled = os.path.join(exp_dir, 'editing_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    print('Load finetuned generator')
    
    target_ckpt = torch.load(checkpoint_path, map_location=device)

    # style_latent = target_ckpt["style_latent"]
    latent_avg = target_ckpt["latent_avg"].type(torch.FloatTensor).to(device)
    # c_dim = target_ckpt['c_dim']
    is_dynagan = target_ckpt['is_dynagan']
    embed_mlp = False
    net = SG2Generator(checkpoint_path, img_size=1024, c_dim=101, no_scaling=False, no_residual=False, is_dynagan=is_dynagan, embed_mlp=embed_mlp).to(device)
    net.eval()
    # conv1.conv.hypernet.out_channel_estimator.weight
    n_latents =  net.generator.n_latent
    # net, opts = load_model(checkpoint_path, update_opts=test_opts)
    # latent_mean=latent_avg.unsqueeze(0).repeat(n_latents,1).unsqueeze(0).repeat(batch_size,1,1)
    
    style_latents = torch.load("/kuacc/users/aanees20/hpc_run/DynaGAN/latents_all.pt").cuda()
    
    clip_model, clip_preprocess = clip.load("ViT-B/16", device = device)

    n_domains = 101
    samples_per_domain = 8
    iters_per_domain = samples_per_domain//batch_size
    
    img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    pbar = tqdm(range(n_domains))

    edit_factor = args.factor #3
    alpha=0.6
    for k in pbar:
        domain = str(k).zfill(3)
        img_num = 0
        output_dir = os.path.join(out_path_results,domain)
        os.makedirs(output_dir,exist_ok=True)

        for i in range(iters_per_domain):
            domain_label, target_pil = get_embed_from_img_name(target_path, domain, clip_preprocess, clip_model)
            domain_label = domain_label#.repeat(4,1)
            target_img_tensor = img_transforms(target_pil).cuda()
            with torch.no_grad():
                sample_w = sample_latents(net, latent_avg)
                sample_w = sample_w + edit_latents * edit_factor
                sample_w[:, 7:, :] = alpha * sample_w[:, 7:, :] + style_latents[k:k+1][:, 7:, :].cuda() * (1-alpha)
                # for edit_dir in
                out_fixed = net([sample_w], input_is_latent=True, randomize_noise=False)[0]
                out = net([sample_w], input_is_latent=True, randomize_noise=False, domain_labels=[domain_label],domain_is_latents=not embed_mlp)[0]
            imgs = [img for img in out]
            # print(target_img_tensor.shape)
            # print(out_fixed.shape)
            # print(imgs[0].shape)
            imgs.insert(0,out_fixed[0])
            imgs.insert(1,target_img_tensor)
            combined = torch.cat(imgs, dim=2)
            combined_pil = tensor2im(combined)
            file_name = str(img_num).zfill(4)+".jpg"
            save_dir = os.path.join(output_dir, file_name)
            combined_pil.save(save_dir)
            img_num +=1


if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_path", type=str, required=True)
    #"/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/latent_editing/"
    parser.add_argument("--factor", type=int, default=1)
    # parser.add_argument("--exp", type=str, default=None, required=True)
    # parser.add_argument("--wandb", action="store_true")
        
    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    
    run(args)
