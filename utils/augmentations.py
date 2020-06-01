import torch
import torch.nn.functional as F
import numpy as np
import random
from skimage import color


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])

    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def hed_jitter(images, targets, theta=0.03):
    alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))
    betti = np.random.uniform(-theta, theta, (1, 3))

    img = (np.array(images) * 255.).astype(np.uint8)
    img = np.transpose(img, [1, 2, 0])

    s = np.reshape(color.rgb2hed(img), (-1, 3))
    ns = alpha * s + betti  # perturbations on HED color space
    nimg = color.hed2rgb(np.reshape(ns, img.shape))

    imin = nimg.min()
    imax = nimg.max()
    rsimg = (((255 * (nimg - imin) / (imax - imin)).astype('uint8')) / 255.).astype(np.float)
    rsimg = torch.from_numpy(np.transpose(rsimg, [2, 0, 1])).type(torch.float32)

    return rsimg, targets


def offline_augment():
    import os
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    def _sync_transform(img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle_list = [0, 90, -90, 180]
            angle = angle_list[random.randint(0, len(angle_list) - 1)]
            img = TF.rotate(img, angle=angle)
            mask = TF.rotate(mask, angle=angle)
        if random.random() < 0.3:  # brightness
            bf_list = np.linspace(0.8, 1.2, 9)
            bf = bf_list[random.randint(0, len(bf_list) - 1)]
            img = TF.adjust_brightness(img, brightness_factor=bf)
        if random.random() < 0.3:  # contrast
            cf_list = np.linspace(0.8, 1.2, 5)
            cf = cf_list[random.randint(0, len(cf_list) - 1)]
            img = TF.adjust_contrast(img, contrast_factor=cf)
        if random.random() < 0.3:  # gamma
            gm_list = np.linspace(0.8, 1.2, 5)
            gm = gm_list[random.randint(0, len(gm_list) - 1)]
            img = TF.adjust_gamma(img, gamma=gm)
        if random.random() < 0.3:
            hf_list = np.linspace(-0.1, 0.1, 11)
            hf = hf_list[random.randint(0, len(hf_list) - 1)]
            img = TF.adjust_hue(img, hue_factor=hf)
        if random.random() < 0.3:
            sf_list = np.linspace(0.8, 1.2, 5)
            sf = sf_list[random.randint(0, len(sf_list) - 1)]
            img = TF.adjust_saturation(img, saturation_factor=sf)

        return img, mask

    data_path = f'../data/classification_patches/positive_patch_pad'
    save_path = f'../data/classification_patches/positive_patch_pad_aug'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for elem in data_list:
        image = Image.open(os.path.join(data_path, elem))
        for times in range(10):
            if times == 0:
                image.save(f'{save_path}/{elem}')
            image, _ = _sync_transform(image, image)
            image.save(f'{save_path}/{times}_{elem}')


if __name__ == '__main__':
    offline_augment()