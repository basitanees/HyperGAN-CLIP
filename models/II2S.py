from functools import partial
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from losses.loss import LossBuilder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# import dlib
import torch
from torch import nn
from models.stylegan2.model import Generator
import numpy as np
import os
from utils.model_utils import download_weight
import torchvision

toPIL = torchvision.transforms.ToPILImage()

class Net(nn.Module):

    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.generator = Generator(opts.size, opts.latent, opts.n_mlp, channel_multiplier=opts.channel_multiplier)
        self.latent_avg = self.generator.mean_latent(10000)[0]
        self.cal_layer_num()
        # self.load_weights()
        print("Loading PCA model")
        self.load_PCA_model()


    def load_weights(self):
        if not os.path.exists(self.opts.ckpt):
            print('Downloading StyleGAN2 checkpoint: {}'.format(self.opts.ckpt))
            download_weight(self.opts.ckpt)

        print('Loading StyleGAN2 from checkpoint: {}'.format(self.opts.ckpt))
        checkpoint = torch.load(self.opts.ckpt)
        device = self.opts.device
        self.generator.load_state_dict(checkpoint['g_ema'], strict=False)
        
        print('Calculating mean latent...')
        # self.latent_avg = checkpoint['latent_avg']
        self.latent_avg = self.generator.mean_latent(10000)
        print('Calculated mean latent!')
        
        self.generator.to(device)
        self.latent_avg = self.latent_avg.to(device)

        print("Setting to eval")
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()
        print("Setting to eval done")


    def build_PCA_model(self, PCA_path):
        print("Building PCA...")
        with torch.no_grad():
            latent = torch.randn((10000, 512), dtype=torch.float32)
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu()
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
            self.generator.style.to(self.opts.device)

        from utils.PCA_utils import IPCAEstimator
        print("Estimating")
        transformer = IPCAEstimator(512)
        X_mean = pulse_space.mean(0)
        transformer.fit(pulse_space - X_mean)
        X_comp, X_stdev, X_var_ratio = transformer.get_components()
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)


    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        if not os.path.isfile(PCA_path):
            download_weight(PCA_path)
            try:
                assert os.path.isfile(PCA_path)
            except AssertionError:
                self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)



    def make_noise(self):
        noises_single = self.generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())
    
        return noises

    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14
        return


    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0).repeat(len(latent_in), 1, 1)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss



class II2S(nn.Module):

    def __init__(self, opts):
        super(II2S, self).__init__()
        self.opts = opts
        self.net = Net(self.opts).cuda()
        self.load_downsampling()
        self.setup_loss_builder()
        # self.set_up_face_predictor()


    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = torch.nn.AdaptiveAvgPool2d((256, 256))#BicubicDownSample(factor=factor)

    def setup_optimizer(self, batch):

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        latent = []
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda().unsqueeze(0).repeat(batch, 1)
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda().unsqueeze(0).repeat(batch, 1)
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer, latent


    def setup_dataloader(self, image_path=None, align_input=False):

        self.dataset = ImagesDataset(opts=self.opts, image_path=image_path,
                                     face_predictor=None, align_input=align_input)
        # self.dataloader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        self.dataloader = DataLoader(self.dataset, batch_size=16, shuffle=False, drop_last=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_loss_builder(self):
        self.loss_builder = LossBuilder(self.opts)


    # def set_up_face_predictor(self):
    #     self.predictor = None
    #     predictor_weight = os.path.join('pretrained_models', 'shape_predictor_68_face_landmarks.dat')
    #     download_weight(predictor_weight)
    #     self.predictor = dlib.shape_predictor(predictor_weight)


    def invert_images(self, image_path=None, output_dir=None, return_latents=False, align_input=False, save_output=True):

        final_latents =None
        if return_latents:
            final_latents = []

        self.setup_dataloader(image_path=image_path, align_input=align_input)
        device = self.opts.device

        for ref_im_H, ref_im_L, ref_name in tqdm(self.dataloader):
            optimizer, latent = self.setup_optimizer(len(ref_im_H))
            pbar = tqdm(range(self.opts.steps), desc='Embedding')
            for step in pbar:
                optimizer.zero_grad()
                latent_in = torch.stack(latent, dim=1)
                gen_im = self.net.generator([latent_in], input_is_latent=True, return_latents=False)[0]
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic.get('p-norm', "NaN")))

                if self.opts.save_intermediate and step % self.opts.save_interval==0 and save_output:
                    self.save_intermediate_results(ref_name, gen_im, latent_in, step, output_dir)

            if save_output:
                self.save_results(ref_name, gen_im, latent_in, output_dir)

            if return_latents:
                final_latents.append(latent_in)

        if return_latents and len(final_latents) != 1:
            final_latents = [torch.cat(final_latents)]
        return final_latents


    def cal_loss(self, im_dict, latent_in):
        loss, loss_dic = self.loss_builder(**im_dict)
        # p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        # loss_dic['p-norm'] = p_norm_loss
        # loss += p_norm_loss

        return loss, loss_dic


    def save_results(self, ref_name, gen_im, latent_in, output_dir):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def save_intermediate_results(self, ref_name, gen_im, latent_in, step, output_dir):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        intermediate_folder = os.path.join(output_dir, ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True