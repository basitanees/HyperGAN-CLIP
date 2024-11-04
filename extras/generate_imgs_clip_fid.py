import torch
import numpy as np
import os
import random
from tqdm import tqdm
import json

from models.DynaGAN import SG2Generator
import torchvision
from datasets.target_dataset_gen import TargetDataset
from torchvision.utils import save_image
from argparse import ArgumentParser
from PIL import Image
import PIL
from losses.clip_loss import CLIPLoss
toPIL = torchvision.transforms.ToPILImage()



def make_label(batch, c_dim, device, label = None):
    c = torch.zeros(batch, c_dim).to(device)
    
    if label is not None:
        c_indicies = [label for _ in range(batch)]
    else:
        c_indicies = torch.randint(0, c_dim, (batch,))
        
    for i, c_idx in enumerate(c_indicies):
        c[i,c_idx] = 1.0
    
    return c

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))#.resize((1024,1024), Image.BICUBIC)

def main(args):
    
    output_imgs_dir = args.output_imgs_dir
    
    output_dir = "/kuacc/users/aanees20/DynaGAN/output"
    ZP_img_clip_embed = torch.from_numpy(np.load(os.path.join(output_dir,"clip_embeds.npy"))).type(torch.FloatTensor)
    
# from losses.clip_loss import CLIPLoss
# device = "cuda"
# clip_models=["ViT-B/16"]
# print("Loading CLIP")
# clip_loss_models = {model_name: CLIPLoss(device,clip_model=model_name) for model_name in clip_models}
# print("Loaded CLIP")
    
    # with open("/kuacc/users/aanees20/DynaGAN/output/args.json") as f:
    #     prev_args=json.load(f)
    # images_paths = prev_args["style_img_dir"]
    # ZP_imgs_clip_embed = []
    # for style_img in images_paths:
    #     ZP_input_img = PIL.Image.open(style_img).convert('RGB')
    #     ZP_input_img_1024 = ZP_input_img#.resize((1024, 1024), PIL.Image.BICUBIC)
    #     ZP_img_tensor = 2.0 * torchvision.transforms.ToTensor()(ZP_input_img_1024).unsqueeze(0).cuda() - 1.0
        
    #     ZP_img_clip_embed = clip_loss_models["ViT-B/16"].encode_images(ZP_img_tensor).to(torch.float32)
    #     ZP_imgs_clip_embed.append(ZP_img_clip_embed)
    # ZP_img_clip_embed = torch.cat(ZP_imgs_clip_embed)#.detach().cpu()
    
    # embeds = ZP_img_clip_embed.detach().cpu().numpy()
    # output_dir = "/kuacc/users/aanees20/DynaGAN/output"
    # latent_path = os.path.join(output_dir, "clip_embeds.npy")
    # np.save(latent_path, embeds)
    
    
    dataset = TargetDataset(ZP_img_clip_embed)
    
#     # Load finetuned generator
    print('Load finetuned generator')

    target_ckpt = torch.load(args.ckpt, map_location=args.device)

    style_latent = target_ckpt["style_latent"]
    latent_avg = target_ckpt["latent_avg"].type(torch.FloatTensor).to(device)
    c_dim = target_ckpt['c_dim']
    is_dynagan = target_ckpt['is_dynagan']

    generator = SG2Generator(args.ckpt, img_size=args.size, c_dim=c_dim, no_scaling=args.no_scaling, no_residual=args.no_residual, is_dynagan=is_dynagan).to(args.device)
    generator.eval()
    n_latents =  generator.generator.n_latent

#     if args.latent_path is None:
#         random_z = torch.randn(args.n_sample, 512).to(args.device)
#     else:
#         random_z = torch.from_numpy(np.load(args.latent_path)).type(torch.FloatTensor).to(args.device)

    batch_size = args.n_sample
    total_imgs = 50000
    iters = int(total_imgs/batch_size/c_dim)+1
    img_idx = 0


    for i in tqdm(range(c_dim)):
        dir_name = f"{output_imgs_dir}/{i}"
        os.makedirs(dir_name, exist_ok=True)
        for j in range(iters):
            with torch.no_grad():
                domain_label = make_label(batch_size, c_dim=c_dim, device=args.device, label=i)
                domain_labels_c = domain_label
                domain_idx = torch.argmax(domain_label, dim=1).tolist()
                domain_label = dataset[domain_idx].cuda()
                domain_idx = torch.argmax(domain_labels_c).item()
                
                random_z = torch.randn(args.n_sample, 512).to(args.device)
                w_styles = generator.style([random_z])[0]

                output_latents = args.truncation * (w_styles - latent_avg) + latent_avg
                output_latents = output_latents.unsqueeze(1).repeat(1, n_latents, 1)

                mixed_latent = output_latents.clone()
                mixed_latent[:, 7:, :] = style_latent[domain_idx:domain_idx+1][:, 7:, :]
                w = [mixed_latent]
                

                
                outputs = generator(w, input_is_latent=True, randomize_noise=False, domain_labels=[domain_label],domain_is_latents=True)[0]

                for image in outputs:
                    img_pil = tensor2im(image)
                    img_pil.save(f"{dir_name}/{img_idx}.jpg")
                    img_idx += 1
                    if img_idx == 50000:
                        break


if __name__ == '__main__':
    device = 'cuda'

    parser = ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=16, help='number of fake images to be sampled')
    parser.add_argument('--n_steps', type=int, default=40, help="determines the granualarity of interpolation")
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="output_disc_conv/checkpoint/final.pt")
    parser.add_argument('--mode', type=str, default='viz_imgs')
    parser.add_argument('--latent_path', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default="samples")
    parser.add_argument('--output_imgs_dir', type=str, default="/kuacc/users/aanees20/Dynagan_eval_data/generated_clip_disc_conv")
    parser.add_argument("--no_scaling",  action='store_true', help="no filter scaling")
    parser.add_argument("--no_residual",  action='store_true', help="no residual scaling")
    parser.add_argument('--each', action='store_true', default=False)

    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.device = "cuda"
    main(args)