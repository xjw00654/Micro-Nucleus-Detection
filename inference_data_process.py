import os
from PIL import Image
from tqdm import tqdm
import numpy as np


def check_and_create_dir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def process(imgs_list, save_path, in_size=2432, out_size=608):
    assert in_size % out_size == 0
    for img_path in imgs_list:
        basename = os.path.basename(img_path).split('.')[0]
        x40_img_pil = Image.open(img_path)
        x20_img_array = np.array(x40_img_pil.resize((in_size // 2, in_size // 2)))
        assert x20_img_array.ndim == 3

        # x20 process loop
        x20_out_path = os.path.join(save_path, 'x20')
        check_and_create_dir(x20_out_path)
        for i in range(x20_img_array.shape[0] // out_size):
            for j in range(x20_img_array.shape[1] // out_size):
                patch = x20_img_array[out_size * i: out_size * (i + 1),
                        out_size * j: out_size * (j + 1), :]
                Image.fromarray(patch).save(os.path.join(x20_out_path, f'{basename}_{i}_{j}.jpg'))

        # x40 process loop
        x40_out_path = os.path.join(save_path, 'x40')
        check_and_create_dir(x40_out_path)
        x40_img_array = np.array(x40_img_pil)
        for i in range(x40_img_array.shape[0] // (out_size * 2)):
            for j in range(x40_img_array.shape[1] // (out_size * 2)):
                patch = x40_img_array[2 * out_size * i: 2 * out_size * (i + 1),
                        2 * out_size * j: 2 * out_size * (j + 1), :]
                Image.fromarray(patch).save(os.path.join(x40_out_path, f'{basename}_{i}_{j}.jpg'))


def get_imgs_list(path, ending='.jpg'):
    imgs_list = []
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if fname.endswith(ending) and not fname.startswith('.'):
                imgs_list.append(os.path.join(root, fname))
    return imgs_list


if __name__ == '__main__':
    basic_path = './data'
    x40_data_path = os.path.join(basic_path, 'MicroNuclei/original_data')
    save_path = os.path.join(basic_path, 'samples')

    data_list = os.listdir(x40_data_path)
    for elem in tqdm(data_list):
        imgs_list = get_imgs_list(os.path.join(x40_data_path, elem))
        elem_save_path = os.path.join(save_path, os.path.basename(elem).split('_')[0])
        process(imgs_list, elem_save_path)
