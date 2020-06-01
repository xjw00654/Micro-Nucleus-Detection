import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def random_elastic_cv2(img, alpha=2, sigma=0.06, mask=None):
    img = np.array(img)
    alpha = img.shape[1] * alpha
    sigma = img.shape[1] * sigma
    if mask is not None:
        mask = np.array(mask).astype(np.uint8)
        img = np.concatenate((img, mask[..., None]), axis=2)

    shape = img.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    # dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
    if mask is not None:
        return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
    else:
        return Image.fromarray(img)


def aug_func(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        angle_list = [0, 90, -90, 180]
        angle = angle_list[random.randint(0, len(angle_list) - 1)]
        img = TF.rotate(img, angle=angle)
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
    # if random.random() < 0.3:
    #     img = random_elastic_cv2(img)
    return img


def augmentation():
    folder_name = 'negative_patch_generated'
    data_path = f'data/classification_patches/data_self_annotated_finetuned_PNG/{folder_name}'
    if not os.path.exists(f'{data_path}_aug'):
        os.mkdir(f'{data_path}_aug')
    data_list = [os.path.join(data_path, elem) for elem in os.listdir(data_path)]

    for idx, img in enumerate(data_list):
        image = Image.open(img)
        for elem in range(5):
            save_path = img.replace(f'{folder_name}/', f'{folder_name}_aug/aug{elem}_')
            aug_image = aug_func(image)
            aug_image.save(save_path)

        print(f'Finished {idx} / {len(data_list)} images.')


if __name__ == '__main__':
    augmentation()
