import torch
from models.II2S import II2S
from options.cat_embed_options import II2S_s_opts

style_img_dir = "/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/wild_data"
ii2s = II2S(II2S_s_opts)

print("Starting inversion...")
latents = ii2s.invert_images(image_path=style_img_dir, output_dir=None,
                            return_latents=True, align_input=False, save_output=False)[0]
print("Ended inversion!")

torch.save(latents, "latents_w_rssa_cats.pt")
