import os
import numpy as np
import random
import shutil
from PIL import Image
import linecache


def padding(mode='train'):
    data_path = f'./data/classification_patches/data_self_annotated_finetuned'
    if mode == 'train':
        data_path = os.path.join(data_path, 'positive_patch')
        data_list = [os.path.join(data_path, elem) for elem in os.listdir(data_path)]
        save_list = [elem.replace('positive_patch', 'positive_patch_pad') for elem in data_list if elem.endswith('.png')]
    else:
        data_path = os.path.join(data_path, 'generated_positive_patch_picked_negative')
        data_list = [os.path.join(data_path, elem) for elem in os.listdir(data_path)]
        save_list = [elem.replace('generated_positive_patch_picked_negative',
                                 'generated_positive_patch_picked_negative_pad') for elem in data_list if elem.endswith('.jpg')]

    target_size = np.array([64, 64])
    for elem, save_elem in zip(data_list, save_list):
        zero_patch = np.zeros(shape=np.append(target_size, 3), dtype=np.uint8)
        image = Image.open(elem)
        image_size_ratio = target_size / image.size
        image = image.resize([int(e) for e in [elem * min(image_size_ratio) for elem in image.size]])

        pos_idx = (target_size - image.size) // 2

        zero_patch[pos_idx[0]:pos_idx[0]+image.size[0], pos_idx[1]:pos_idx[1]+image.size[1], :] = np.transpose(np.array(image), [1, 0 ,2])
        Image.fromarray(zero_patch).save(f'{save_elem}')


def get_negative_patch():
    data_path = '/home/jwxie/Desktop/Micro-Nucleu/PyTorch-YOLOv3/data'
    save_path = '/home/jwxie/Desktop/negative_patch/'

    folder_list = [e for e in os.listdir(data_path) if e.startswith('yolo_cls_data')]
    for folder in folder_list:
        data_path_full = os.path.join(data_path, folder)
        data_list = os.listdir(data_path_full)
        for img in data_list:
            if random.random() < 0.2:
                src = os.path.join(data_path_full, img)
                des = os.path.join(save_path, img)
                shutil.copy(src, des)


if __name__ == '__main__':
    padding(mode='train')
    padding(mode='validation')

