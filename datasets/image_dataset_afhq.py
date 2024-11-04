from torch.utils.data import Dataset
from PIL import Image
import PIL
from utils import data_utils
import torchvision.transforms as transforms
import os
from utils.shape_predictor import align_face
import sys
import glob

class ImagesDataset(Dataset):

    def __init__(self, opts, image_path=None):

        if type(image_path) == list:
            self.image_paths = image_path
        elif os.path.isdir(image_path):
            self.image_paths = glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png")) + glob.glob(os.path.join(image_path, "*.jpeg"))
            # self.image_paths = sorted(data_utils.make_dataset(image_path))
        elif os.path.isfile(image_path):
            self.image_paths = [image_path]
        else:
            sys.exit('Invalid Input')


        self.image_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.opts = opts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_path = self.image_paths[index]

        im_H = Image.open(im_path).convert('RGB')
        if im_H.size[0] != 1024:
            im_H = im_H.resize((512, 512), PIL.Image.BICUBIC)
        im_L = im_H.resize((256, 256), PIL.Image.LANCZOS)
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        if self.image_transform:
            im_H = self.image_transform(im_H)
            im_L = self.image_transform(im_L)

        return im_H, im_L, im_name


# class ImagesDataset(Dataset):

# 	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
# 		self.source_paths = sorted(data_utils.make_dataset(source_root))
# 		self.source_transform = source_transform
# 		self.opts = opts

# 	def __len__(self):
# 		return len(self.source_paths)

# 	def __getitem__(self, index):
# 		from_path = self.source_paths[index]
# 		from_im = Image.open(from_path)
# 		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

# 		if self.source_transform:
# 			from_im = self.source_transform(from_im)

# 		return from_im