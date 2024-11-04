
import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from pathlib import Path
from models.DynaGAN_afhq import DynaGAN
from utils.file_utils import save_images
from options.DynaGAN_options_afhq import DynaGANOptions
from datasets.image_dataset_afhq import ImagesDataset
toPIL = torchvision.transforms.ToPILImage()
import vision_aided_loss
import random
import wandb
from PIL import Image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


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
    return Image.fromarray(var.astype('uint8'))

def train(args, output_dir):
    # Set up networks, optimizers.
    print("Initializing networks...")
    args.n_sample = 7
    # args.image_path = "/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/nada_data_2/"
    embed_mlp = False
    use_disc = True

    print("Loading DynaGAN...")
    net = DynaGAN(args)
    print("Loaded DynaGAN...")
    # style_latent = net.embed_style_img(args.style_img_dir)
    # args.iter = 100
    
    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr = args.lr,
        betas = (0, 0.99)
    )
    
    if use_disc:
        print("Load discriminator...")
        discr = vision_aided_loss.Discriminator(cv_type='clip', loss_type='multilevel_sigmoid_s', device=args.device, output_type='conv_multi_level').to(args.device) #output_type='conv_multi_level''conv_multi_level'
        discr.cv_ensemble.requires_grad_(False) # Freeze feature extractor
        print("Loaded discriminator...")
    
        d_optim = torch.optim.Adam(
            params=discr.decoder.parameters(), lr=args.lr,
            betas = (0, 0.99), weight_decay=0, eps=1e-8)
        
    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
    
    sample_dir = os.path.join(output_dir, "sample")
    ckpt_dir = os.path.join(output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Set fixed sample
    fixed_z = torch.randn(args.n_sample, 512, device=args.device)
    with torch.no_grad():
        fixed_w_styles = net.generator_frozen.style([fixed_z])
        orig_image_H = net.generator_frozen(fixed_w_styles, input_is_latent=True, truncation=0.9)[0]
        # fixed_embed = net.clip_loss_models["ViT-B/16"].encode_images(orig_image_H).to(torch.float32)
    orig_L = face_pool(orig_image_H)
    
    
    dataset = ImagesDataset(args, args.style_img_dir)
    
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    
    dataloader_sample = DataLoader(dataset, batch_size=args.n_sample, shuffle=False, drop_last=False)
    target_H, target_L, _ = next(iter(dataloader_sample))
    target_H, target_L = target_H.cuda(), target_L.cuda()
    with torch.no_grad():
        target_embed = net.clip_loss_models["ViT-B/16"].encode_images(target_H).to(torch.float32)
        target_L = face_pool(target_H)
    # np.save("test_latent.npy",fixed_z.cpu().numpy())
    # save_image(net, fixed_z, args, sample_dir, -1)

    config = vars(args)
    wandb.init(project="Domain-adaptation",config=config)
    print("Wandb init")
    
    # Training loop
    pbar = tqdm(range(args.iter))
    i =0
    for j in pbar:
        # pbar = tqdm(enumerate(dataloader))
        for batch_target in dataloader:
    # for i, batch_target in pbar:
            if use_disc:
                d_optim.zero_grad()
                net.zero_grad()
                
            net.train()

            sample_z = torch.randn(args.batch, 512, device=args.device)

            # sample_domain_label = make_label(args.batch, c_dim=args.c_dim, device=args.device)
            real = batch_target[0].cuda()
            real_L = batch_target[1].cuda()
            
            # select same image 25% of the times
            # indices = torch.bernoulli(torch.ones(args.batch)*0.75)[:,None,None,None].cuda()
            # if 0 in indices:
            #     with torch.no_grad():
            #         w_styles = net.generator_frozen.style([sample_z])
            #         same_image_H = net.generator_frozen(w_styles, input_is_latent=True, truncation=1)[0]
            #         real = torch.where(indices==1, real, same_image_H)
            #         same_image_L = net.D_VGG(same_image_H)
            #         real_L = torch.where(indices==1, real_L, same_image_L)
            
            
            if args.use_truncation_in_training:
                [fake, _, embeds, loss_dict], loss = net([sample_z], truncation=args.sample_truncation, target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
            else:
                [fake, _, embeds, loss_dict], loss = net([sample_z], target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
                
            
            if use_disc:
                real_embed = net.clip_loss_models["ViT-B/16"].encode_images(real).to(torch.float32)
                fake_embed = net.clip_loss_models["ViT-B/16"].encode_images(fake).to(torch.float32)
                # new_img, ZP_img_clip_embed = new_img.cuda(), ZP_img_clip_embed.cuda()
                # domain_labels = [ZP_img_clip_embed]

                lossD = discr(real, c=real_embed, for_real=True) + discr(fake, c=fake_embed, for_real=False) + discr(real, c=fake_embed, for_real=False) + discr(fake, c=real_embed, for_real=False)
                lossD = lossD.mean() * 0.2
                lossD.backward()
                d_optim.step()
            
            net.zero_grad()
            g_optim.zero_grad()
            
            loss_dict_log = loss_dict
            loss_dict_log["DynaGAN loss"] = loss.item()
            
            if use_disc:
                if args.use_truncation_in_training:
                    [fake, _, embeds, _], loss = net([sample_z], truncation=args.sample_truncation, target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
                else:
                    [fake, _, embeds, _], loss = net([sample_z], target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
                
                fake_embed = net.clip_loss_models["ViT-B/16"].encode_images(fake).to(torch.float32)
                lossG = discr(fake, c=fake_embed, for_G=True)
                lossG = lossG.mean() * 0.2
            else:
                lossG = 0.0
            loss_G = loss + lossG
            loss_G.backward()
            
            g_optim.step()
            
            if use_disc:
                pbar.set_description(f"Training | DynaGAN loss: {loss.item():.3f} | Gen loss: {loss_G.item():.3f} | Dis loss: {lossD.item():.3f}")
                loss_dict_log.update({"Generator_loss": loss_G.item(), "Discriminator_loss": lossD.item()})
            else:
                pbar.set_description(f"Training | DynaGAN loss: {loss.item():.3f}")
                   
            
            wandb.log(loss_dict_log)
            # net.zero_grad()
            # loss.backward()
            # g_optim.step()
            
            # pbar.set_description(f"Finetuning Generator | Total loss: {loss}")

            if (i % args.vis_interval == 0 or (i + 1) == args.iter):
                orig = orig_L
                target = target_L
                target = face_pool(target)
                with torch.no_grad():
                    generated = face_pool(net.generator_trainable(fixed_w_styles, input_is_latent=True, randomize_noise=False, domain_labels=[target_embed],domain_is_latents=True)[0])
                # print(orig.shape)
                # print(generated.shape)
                # print(target.shape)
                combined = torch.cat([orig,generated,target],dim=0)
                grid = make_grid(combined, nrow=args.n_sample)
                images = tensor2im(grid)
                
                images = wandb.Image(images, caption="Top: original, Middle: Conditioned, Bottom: Target")
                wandb.log({"Generate image logs": images})
                
                
                target = target_L
                with torch.no_grad():
                    # print(target_L.shape)
                    target_projected_w_res = net.encoder(target)
                    # print(target_projected_w_res.shape)
                    target_projected_w_res[:,:7,:] *= 0.8
                    target_projected_w_res[:,7:,:] *= 0.8
                    target_projected_w = net.generator_frozen.mean_latent[None,:] + target_projected_w_res
                    target_recon = face_pool(net.generator_frozen([target_projected_w], input_is_latent=True, truncation=1., randomize_noise=False)[0])
                    target_recon_cond = face_pool(net.generator_trainable([target_projected_w], input_is_latent=True, randomize_noise=False, domain_labels=[target_embed],domain_is_latents=True)[0])
                
                orig = target_recon
                generated = target_recon_cond
                combined = torch.cat([orig,generated,target],dim=0)
                grid = make_grid(combined, nrow=args.n_sample)
                images_2 = tensor2im(grid)
                
                images_log = wandb.Image(images_2, caption="Top: original, Middle: Conditioned, Bottom: Target")
                wandb.log({"Training Image Reconstruction": images_log})
                    
            # if (i % args.vis_interval == 0 or (i + 1) == args.iter):
            #     save_image(net, fixed_z, args, sample_dir, i, batch_target, not embed_mlp)

            if args.save_interval is not None and ((i + 1) % args.save_interval == 0 or (i + 1) == args.iter):
                ckpt_name = '{}/{}.pt'.format(ckpt_dir, str(i + 1).zfill(6))
                save_checkpoint(net, g_optim, ckpt_name, embed_mlp=embed_mlp)
            
            i+=1

    ckpt_name = '{}/{}.pt'.format(ckpt_dir, "final")
    save_checkpoint(net, g_optim, ckpt_name, human_face=args.human_face, embed_mlp=embed_mlp)




def save_image(net, fixed_z, args, sample_dir, i, batch_target, domain_is_latents):
    net.eval()
    with torch.no_grad():
        for domain_idx in range(args.c_dim):
            # domain_label = make_label(args.n_sample, c_dim=args.c_dim, device=args.device, label=domain_idx)
            [sampled_src, sampled_dst, rec_dst, without_color_dst], loss = net([fixed_z],
                                                                        truncation=args.sample_truncation, 
                                                                        target_img_H=batch_target[0], 
                                                                        target_img_L=batch_target[1],
                                                                        inference=True,
                                                                        domain_is_latents=domain_is_latents)
            grid_rows = int(args.n_sample ** 0.5)
            save_images(sampled_dst, sample_dir, f"dst_{domain_idx}", grid_rows, i+1)
            save_images(without_color_dst, sample_dir, f"without_color_{domain_idx}", grid_rows, i+1)
            save_images(rec_dst, sample_dir, f"rec_{domain_idx}", grid_rows, i+1)
        
        
def save_checkpoint(net, g_optim, ckpt_name, human_face=False, is_dynagan=True, embed_mlp=False):
    print(f"Save the checkpoint! {ckpt_name}")
    save_dict = {
            "g_ema": net.generator_trainable.generator.state_dict(),
            "g_optim": g_optim.state_dict(),
            "latent_avg": net.generator_trainable.mean_latent,
            "is_dynagan": is_dynagan,
            "human_face": human_face,
            "embed_mlp": embed_mlp
        }
    
    torch.save(
        save_dict, 
        ckpt_name
    )


if __name__ == "__main__":

    option = DynaGANOptions()
    parser = option.parser

    # I/O arguments
    parser.add_argument('--style_img_dir', type=str, default="/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/wild_data",
                        help='Style image')
    parser.add_argument('--output_dir', type=str, default="output_cats_to_dogs_wild_dyna")
    parser.add_argument("--lambda_contrast", type=float, default=1.0, help="Weight of contrastive loss")
    parser.add_argument("--lambda_id", type=float, default=3.0, help="Weight of identity loss") # 3.0
    parser.add_argument("--load_inverted_latents",  action='store_true', help="load inverted latents")
    parser.add_argument("--no_scaling",  action='store_true', help="no filter scaling")
    parser.add_argument("--no_residual",  action='store_true', help="no residual scaling")
    parser.add_argument("--id_model_path",  type=str, default="pretrained_models/moco_v2_800ep_pretrain.pt", help="identity path")
    parser.add_argument("--human_face", action='store_true', help="Whether it is for human faces", default=False)
    parser.add_argument("--stylegan_size", type=int, default=512, help="generator size")
    parser.add_argument('--encoder_type', type=str, default="Encoder4Editing")
    parser.add_argument('--e4e_checkpoint_path', type=str, default="/kuacc/users/aanees20/hpc_run/pretrained_models/best_model_cats.pt")
    # parser.add_argument("--iter", type=int, default=10000, help="num epochs")
    
    args = option.parse()
    
    args.style_img_dir = glob.glob(os.path.join(args.style_img_dir, "*.jpg")) + glob.glob(os.path.join(args.style_img_dir, "*.png")) + glob.glob(os.path.join(args.style_img_dir, "*.jpeg"))
    args.c_dim = len(args.style_img_dir)
    args.save_interval = 500
    print(f"Number of domains: {args.c_dim}")
    output_dir = args.output_dir # os.path.join(args.output_dir, Path(args.style_img).stem)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    import time

    start_time = time.time()
    train(args, output_dir)
    end_time = time.time()
    print(f"Training time {end_time-start_time}s")
