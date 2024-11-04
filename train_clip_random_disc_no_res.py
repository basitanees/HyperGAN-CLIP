
import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from models.DynaGAN_ffhq import DynaGAN
from utils.file_utils import save_images
from options.DynaGAN_options_ffhq import DynaGANOptions
toPIL = torchvision.transforms.ToPILImage()
import vision_aided_loss
import random
import wandb
from PIL import Image
from torchvision.utils import make_grid


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
    
    # image_path = "/datasets/ffhq/images1024x1024/"
    embed_mlp = False
    
    # dataset = ImagesDataset(args, image_path)
    # dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    net = DynaGAN(args)
    # style_latent = net.embed_style_img(args.style_img_dir)
    
    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr = args.lr,
        betas = (0, 0.99)
    )
    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
    
    discr = vision_aided_loss.Discriminator(cv_type='clip', loss_type='multilevel_sigmoid_s', device=args.device, output_type='conv_multi_level').to(args.device) #output_type='conv_multi_level''conv_multi_level'
    discr.cv_ensemble.requires_grad_(False) # Freeze feature extractor
    
    d_optim = torch.optim.Adam(
        params=discr.decoder.parameters(), lr=args.lr,
        betas = (0, 0.99), weight_decay=0, eps=1e-8)
    
    
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
        orig_image_H = net.generator_frozen(fixed_w_styles, input_is_latent=True, truncation=1)[0]
        fixed_embed = net.clip_loss_models["ViT-B/16"].encode_images(orig_image_H).to(torch.float32)
    orig_L = net.D_VGG(orig_image_H)
    
    indices = torch.ones(args.batch)
    indices[-1] = 0
    indices = indices[:,None,None,None].cuda()
    
    target_z = torch.randn(args.n_sample, 512, device=args.device)
    with torch.no_grad():
        target_w_styles = net.generator_frozen.style([target_z])
        target_image_H = net.generator_frozen(target_w_styles, input_is_latent=True, truncation=1)[0]
        target_H = torch.where(indices==1, target_image_H, orig_image_H)
        target_embed = net.clip_loss_models["ViT-B/16"].encode_images(target_H).to(torch.float32)
        
    target_L = net.D_VGG(target_H)
    # target_L = torch.where(indices==1, target_image_L, orig_L)
    
    np.save("test_latent.npy",fixed_z.cpu().numpy())
    # save_image(net, fixed_z, args, sample_dir, -1)

    # Training loop
    # pbar = tqdm(range(args.iter))
    # for i in pbar:
    args.iter = 15000
    
    # Initialize wandb for monitoring
    config = vars(args)
    wandb.init(project="CLIP2StyleGAN",config=config)
    
    
    pbar = tqdm(range(args.iter))
    i =0
    for i in pbar:
        d_optim.zero_grad()
        net.zero_grad()
        
        net.train()

        sample_z = torch.randn(args.batch, 512, device=args.device)
        sample_z2 = torch.randn(args.batch, 512, device=args.device)

        # sample_domain_label = make_label(args.batch, c_dim=args.c_dim, device=args.device)
        # real = batch_target[0].cuda()
        # real_L = batch_target[1].cuda()
        # if i < 1000:
        #     alpha = (random.random() * 0.25)
        # elif i < 2000:
        #     alpha = (random.random() * 0.5)
        # elif i < 3000:
        #     alpha = (random.random() * 0.75)
        # else:
        alpha = random.random()
        # select same image 25% of the times
        # indices = torch.bernoulli(torch.ones(args.batch)*0.75)[:,None,None,None].cuda()
        with torch.no_grad():
            w_styles = net.generator_frozen.style([sample_z])
            # print(w_styles)
            w_styles2 = net.generator_frozen.style([sample_z2])
            w_styles2 = [(w_styles2[0] * alpha + w_styles[0] * (1-alpha))] # bring the target images closer to the source by a random amount
            same_image_H = net.generator_frozen(w_styles, input_is_latent=True, truncation=1)[0]
            same_embed = net.clip_loss_models["ViT-B/16"].encode_images(same_image_H).to(torch.float32)
            same_image_H2 = net.generator_frozen(w_styles2, input_is_latent=True, truncation=1)[0]
            # real = torch.where(indices==1, same_image_H2, same_image_H)
            real = same_image_H2
            # same_image_L = net.D_VGG(same_image_H)
            same_image_L2 = net.D_VGG(same_image_H2)
            # real_L = torch.where(indices==1, same_image_L2, same_image_L)
            real_L = same_image_L2
        
        
        if args.use_truncation_in_training:
            [fake, _, _, loss_dict], loss = net([sample_z], truncation=args.sample_truncation, target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
        else:
            [fake, _, _, loss_dict], loss = net([sample_z], target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
        
        real_embed = net.clip_loss_models["ViT-B/16"].encode_images(real).to(torch.float32)
        fake_embed = net.clip_loss_models["ViT-B/16"].encode_images(fake).to(torch.float32)
        # new_img, ZP_img_clip_embed = new_img.cuda(), ZP_img_clip_embed.cuda()
        # domain_labels = [ZP_img_clip_embed]

        lossD = discr(real, c=real_embed*0, for_real=True) + discr(fake, c=fake_embed-same_embed, for_real=False) + discr(real, c=fake_embed-real_embed, for_real=False) + discr(fake, c=real_embed-same_embed, for_real=False)
        lossD = lossD.mean() * 0.2
        lossD.backward()
        d_optim.step()
        
        net.zero_grad()
        g_optim.zero_grad()
        
        loss_dict_log = loss_dict
        loss_dict_log["DynaGAN loss"] = loss.item()
        
        if args.use_truncation_in_training:
            [fake2, _, _, loss_dict], loss = net([sample_z], truncation=args.sample_truncation, target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
        else:
            [fake2, _, _, loss_dict], loss = net([sample_z], target_img_H=real, target_img_L=real_L, domain_is_latents=not embed_mlp)
        
        fake_embed = net.clip_loss_models["ViT-B/16"].encode_images(fake2).to(torch.float32)
        lossG = discr(fake2, c=fake_embed-same_embed, for_G=True)
        lossG = lossG.mean() * 0.2
        loss_G = loss + lossG
        loss_G.backward()
        
        g_optim.step()
        
        pbar.set_description(f"Training | DynaGAN loss: {loss.item():.3f} | Gen loss: {loss_G.item():.3f} | Dis loss: {lossD.item():.3f}")
        
        loss_dict_log.update({"Generator_loss": loss_G.item(), "Discriminator_loss": lossD.item()})
        wandb.log(loss_dict_log)
        # net.generator_trainable.generator.convs[1].conv.residual_multiplier
        
        # net.zero_grad()
        # loss.backward()
        # g_optim.step()
        
        # pbar.set_description(f"Finetuning Generator | Total loss: {loss}")

        
        if (i % args.vis_interval == 0 or (i + 1) == args.iter):
            orig = orig_L
            target = target_L
            with torch.no_grad():
                generated = face_pool(net.generator_trainable([fixed_w_styles], input_is_latent=True, randomize_noise=False, domain_labels=[target_embed-fixed_embed],domain_is_latents=True)[0])
            combined = torch.cat([orig,generated,target],dim=0)
            grid = make_grid(combined, nrow=args.batch)
            images = tensor2im(grid)
            
            images = wandb.Image(images, caption="Top: original, Middle: Conditioned, Bottom: Target")
            wandb.log({"Generate image logs": images})
        
        # if (i % args.vis_interval == 0 or (i + 1) == args.iter):
        #     save_image(net, fixed_z, args, sample_dir, i, batch_target, not embed_mlp)

        if args.save_interval is not None and ((i + 1) % args.save_interval == 0 or (i + 1) == args.iter):
            ckpt_name = '{}/{}.pt'.format(ckpt_dir, str(i + 1).zfill(6))
            save_checkpoint(net, g_optim, ckpt_name, embed_mlp=embed_mlp)

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
    parser.add_argument('--style_img_dir', type=str, default="target_data/raw_data",
                        help='Style image')
    parser.add_argument('--output_dir', type=str, default="output_clip")
    parser.add_argument("--lambda_contrast", type=float, default=1.0, help="Weight of contrastive loss")
    parser.add_argument("--lambda_id", type=float, default=5, help="Weight of identity loss") # 3.0
    parser.add_argument("--load_inverted_latents",  action='store_true', help="load inverted latents")
    parser.add_argument("--no_scaling",  action='store_true', help="no filter scaling")
    parser.add_argument("--no_residual",  action='store_true', help="no residual scaling")
    parser.add_argument("--id_model_path",  type=str, default="pretrained_models/model_ir_se50.pth", help="identity path")
    parser.add_argument("--human_face", action='store_true', help="Whether it is for human faces", default=False)
    parser.add_argument("--stylegan_size", type=int, default=1024, help="generator size")
    parser.add_argument('--encoder_type', type=str, default="Encoder4Editing")
    parser.add_argument('--e4e_checkpoint_path', type=str, default="/kuacc/users/aanees20/DynaGAN/pretrained_models/e4e_ffhq_encode.pt")
    args = option.parse()
    
    args.style_img_dir = glob.glob(os.path.join(args.style_img_dir, "*.jpg")) + glob.glob(os.path.join(args.style_img_dir, "*.png")) + glob.glob(os.path.join(args.style_img_dir, "*.jpeg"))
    args.c_dim = len(args.style_img_dir)
    args.save_interval = 1000
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
