from torch.utils.data import Dataset

class TargetDataset(Dataset):

    def __init__(self,ZP_img_clip_tensor):
        self.ZP_img_clip_tensor = ZP_img_clip_tensor

    def __len__(self):
        return len(self.ZP_img_clip_tensor)

    def __getitem__(self, index):
        ZP_img_clip_tensor = self.ZP_img_clip_tensor[index]
        return ZP_img_clip_tensor



