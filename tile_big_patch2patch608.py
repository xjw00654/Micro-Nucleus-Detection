import os
import numpy as np
from PIL import Image


if __name__ == '__main__':
    image_list = []
    for root, _, fnames in os.walk('./data/MicroNuclei/original_data/284_37_20'):
        for fname in fnames:
            if fname.endswith('.jpg'):
                img = np.array(Image.open(os.path.join(root, fname)))
                for i in range(2):
                    for j in range(2):
                        patch = img[i * 608:(i + 1) * 608, j * 608:(j + 1) * 608, :]
                        patch_name = fname.replace('.jpg', f'_{i}_{j}.jpg')
                        Image.fromarray(patch).save(f'./data/samples/{patch_name}')