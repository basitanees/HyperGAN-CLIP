import torch
from models.stylegan2.model import Generator
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm

def style(generator, styles):
    '''
    Convert z codes to w codes.
    '''
    styles = [generator.style(s) for s in styles]
    return styles

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8')).resize((1024,1024), Image.BICUBIC)

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

print('Load finetuned generator')
generator = Generator(256, 512, 8).to("cuda")
checkpoint = torch.load('pretrained_models/ffhq_sketches.pt')
gen_weights = checkpoint['g_ema']
generator.load_state_dict(get_keys(gen_weights, "module"), strict=True)

# sample_c = torch.zeros(8, dtype=torch.long, device="cuda")
img_idx = 0
total_imgs = 900
batch_size = 16
iters = int(total_imgs/batch_size)+1

for i in tqdm(range(iters)):
    sample_z = torch.randn(batch_size, 512, device="cuda")
    w_styles = generator.style(sample_z)
    with torch.no_grad():
        img, _ = generator([w_styles], input_is_latent=True, truncation=1, randomize_noise=True)

    for image in img:
        img_pil = tensor2im(image)
        img_pil.save(f"/kuacc/users/aanees20/Dynagan_eval_data/original/sketch/{img_idx}.jpg")
        img_idx += 1

        if img_idx == 900:
            break
print("Done")