# from models.DynaGAN_ffhq import DynaGAN
# from models.DynaGAN_nada import SG2Generator

# from models.DynaGAN_nada import SG2Generator #as SG2GeneratorCLIP
from models.DynaGAN_nada_res import SG2Generator #as SG2GeneratorCLIPRes

import numpy as np
import torch
import torchvision
import os

toPIL = torchvision.transforms.ToPILImage()
from torchvision import transforms
from PIL import Image

from torch import nn
import clip
from tqdm import tqdm
import PIL

# exp_name = "CLIP_base"
# outputs = f"/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/Ablations_quant/{exp_name}/"#####
# ckpt_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_101_no_discc_no_res/checkpoint/final.pt"


# exp_name = "CLIP_disc"
# outputs = f"/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/Ablations_quant/{exp_name}/"#####
# ckpt_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_101_disc_no_res/checkpoint/010000.pt"


exp_name = "CLIP_res"
outputs = f"/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/Ablations_quant/{exp_name}/"#####
ckpt_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_101_no_discc/checkpoint/018000.pt"


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def get_embed_from_img_name(target_path, img_name, clip_preprocess, clip_model):
    batch_size = 8
    img1 = os.path.join(target_path, img_name)
    img1 = f"{img1}/{img_name}.jpg"
    img_cond_1 = Image.open(img1)
    img_preprocess_1 = clip_preprocess(img_cond_1).unsqueeze(0).cuda()
    with torch.no_grad():
        img_embed_1 = clip_model.encode_image(img_preprocess_1).to(torch.float32)
    img_embed_1 = img_embed_1.repeat(batch_size,1) ###
    return img_embed_1#, img_preprocess_1

def sample_latents(args, generator, latent_avg, n_latents, save=False):
    sample_z = torch.randn(args.batch, 512, device=args.device)
    w_styles = generator.style([sample_z])[0]
    output_latents = args.truncation * (w_styles - latent_avg) + latent_avg
    output_latents = output_latents.unsqueeze(1).repeat(1, n_latents, 1)
    if save:
        torch.save(sample_z, "z_frozen.pt")
        torch.save(output_latents, "w_frozen.pt")
    return output_latents

class DynaGANOptions:
    device = "cuda"
    frozen_gen_ckpt = "/kuacc/users/aanees20/hpc_run/DynaGAN/pretrained_models/ffhq.pt"
    size = 1024
    stylegan_size = 1024
#     train_gen_ckpt = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_101_no_res_2/checkpoint/012000.pt"
    # train_gen_ckpt = "/kuacc/users/aanees20/hpc_run/DynaGAN/output_clip_nada_4/checkpoint/011500.pt"
    c_dim = 9
    no_scaling = False
    no_residual = False
    phase = None
    e4e_checkpoint_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/pretrained_models/e4e_ffhq_encode.pt"
    lambda_id = 0
    clip_models = ["ViT-B/32", "ViT-B/16"]
    clip_model_weights = [1.0, 1.0]
    lambda_direction = 1.0
    lambda_patch = 0.0
    lambda_global = 0.0
    lambda_manifold = 0.0
    lambda_texture = 0.0
    lambda_contrast = 0
    batch = 8
    truncation = 0.7
    style_img_dir = "/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/nada_data_2/"
    

args = DynaGANOptions
args.train_gen_ckpt = ckpt_path

print('Load finetuned generator')

target_ckpt = torch.load(args.train_gen_ckpt, map_location=args.device)

# style_latent = target_ckpt["style_latent"]
latent_avg = target_ckpt["latent_avg"].type(torch.FloatTensor).to(args.device)
# c_dim = target_ckpt['c_dim']
is_dynagan = target_ckpt['is_dynagan']

################
embed_mlp = False#########
################
generator = SG2Generator(args.train_gen_ckpt, img_size=args.size, c_dim=101, no_scaling=args.no_scaling, no_residual=args.no_residual, is_dynagan=is_dynagan, embed_mlp=embed_mlp).to(args.device)
generator.eval()
# conv1.conv.hypernet.out_channel_estimator.weight
n_latents =  generator.generator.n_latent

latent_mean=latent_avg.unsqueeze(0).repeat(n_latents,1).unsqueeze(0).repeat(args.batch,1,1)

img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

upsample = nn.Upsample(scale_factor=7)
avg_pool = nn.AvgPool2d(kernel_size=1024 // 32)
face_pool = torch.nn.AdaptiveAvgPool2d((1024, 1024))

clip_model, clip_preprocess = clip.load("ViT-B/16", device = args.device)

root_path = "/scratch/users/aanees20/hpc_run/DynaGAN/target_data"
target = "nada_data_folder_wise/"
target_path = os.path.join(root_path, target)

names = os.listdir(target_path)

seed = 2 #@param {"type": "integer"}

torch.manual_seed(seed)
np.random.seed(seed)

fixed_z = torch.randn(4, 512, device=args.device)

n_domains = 101
samples_per_domain = 600
iters_per_domain = samples_per_domain//args.batch

style_img_dir = []
for i in range(101):
    domain = str(i).zfill(3)
    path = os.path.join(target_path, domain, domain+".jpg")
    style_img_dir.append(path)

ZP_target_latent = torch.load("/kuacc/users/aanees20/hpc_run/DynaGAN/latents_all.pt").cuda()

pbar = tqdm(range(n_domains))
alpha = 0.6

for k in pbar:
    domain = str(k).zfill(3)
    img_num = 0
    output_dir = os.path.join(outputs,domain)
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(iters_per_domain):
        domain_label = get_embed_from_img_name(target_path, domain, clip_preprocess, clip_model) #move outside loop?
        with torch.no_grad():
            sample_w = sample_latents(args, generator, latent_avg, n_latents)
            sample_w[:, 7:, :] = alpha * sample_w[:, 7:, :] + ZP_target_latent[k:k+1][:, 7:, :].cuda() * (1-alpha)
            out = generator([sample_w], input_is_latent=True, randomize_noise=False, domain_labels=[domain_label],domain_is_latents=not embed_mlp)[0]

        imgs = [img for img in out]
        for img in imgs:
            file_name = str(img_num).zfill(4)+".jpg"
            pil_img = tensor2im(img)
            pil_img.save(os.path.join(output_dir,file_name))
            img_num +=1

            