import os
import re
import linecache
import numpy as np
from pathlib import Path
from PIL import Image
import random


def create_positive_patch():
    data_path = Path(__file__).parent / 'data/MicroNuclei/annotated_data/'
    annotation_path = data_path / 'label'
    image_path = data_path / 'images'

    annotation_list = annotation_path.glob('*.txt')
    for one_annotation_path in annotation_list:
        annotation = linecache.getlines(str(one_annotation_path))
        try:
            image = Image.open(str(image_path / one_annotation_path.with_suffix('.jpg').name))
        except FileNotFoundError:
            print(f'Skipped {one_annotation_path.name} for could not find its JPEG image file.')
            continue
        for idx, one_line in enumerate(annotation):
            _, center_x, center_y, w, h = one_line.strip().split(' ')
            start_x = int((float(center_x) - float(w) / 2) * image.size[0])
            start_y = int((float(center_y) - float(h) / 2) * image.size[1])

            end_x = int((float(center_x) + float(w) / 2) * image.size[0])
            end_y = int((float(center_y) + float(h) / 2) * image.size[1])

            # patch = image.crop((start_x, start_y, end_x, end_y))
            ps, margin_x, margin_y = [end_x - start_x, end_y - start_y], 0, 0
            if ps[0] < 64 and ps[1] < 64:
                margin_x = 64 - ps[0]
                margin_y = 64 - ps[1]
            elif ps[0] > 64 or ps[1] > 64:
                margin_x = max(ps) - ps[0]
                margin_y = max(ps) - ps[1]

            start_x -= margin_x // 2
            start_y -= margin_y // 2
            end_x += margin_x // 2
            end_y += margin_y // 2

            patch = image.crop((start_x, start_y, end_x, end_y)).resize((64, 64))
            assert patch.size == (64, 64)

            patch.save(
                Path(__file__).parent / 'data' / 'classification_patches' /
                'data_self_annotated_finetuned_PNG' / 'positive' / f'{one_annotation_path.name[:-4]}_{idx}.png')


def create_negative_patch():
    data_path = Path(__file__).parent / 'data/custom'
    annotation_path = data_path / 'labels'
    image_path = data_path / 'images'

    annotation_list = annotation_path.glob('Normal*.txt')
    for one_annotation_path in annotation_list:
        annotation = linecache.getlines(str(one_annotation_path))
        try:
            file_name = one_annotation_path.name
            spl = list(map(int, re.compile('_\d*_\d*').findall(file_name)[0].split('_')[1:]))
            # image = Image.open(image_path / f'batch{spl[0]}' / f'Normal-3_{spl[0]}_{spl[1]}.jpg')
            image = Image.open(image_path / f'Normal-3_{spl[0]}_{spl[1]}.jpg')
        except FileNotFoundError:
            print(f'Skipped {one_annotation_path.name} for could not find its JPEG image file.')
            continue
        for idx, one_line in enumerate(annotation):
            _, center_x, center_y, w, h = one_line.strip().split(' ')
            start_x = int((float(center_x) - float(w) / 2) * image.size[0])
            start_y = int((float(center_y) - float(h) / 2) * image.size[1])

            end_x = int((float(center_x) + float(w) / 2) * image.size[0])
            end_y = int((float(center_y) + float(h) / 2) * image.size[1])

            # patch = image.crop((start_x, start_y, end_x, end_y))
            ps, margin_x, margin_y = [end_x - start_x, end_y - start_y], 0, 0
            if ps[0] < 32 and ps[1] < 32:
                margin_x = 32 - ps[0]
                margin_y = 32 - ps[1]
            elif ps[0] > 32 or ps[1] > 32:
                margin_x = max(ps) - ps[0]
                margin_y = max(ps) - ps[1]

            start_x -= margin_x // 2
            start_y -= margin_y // 2
            end_x += margin_x // 2
            end_y += margin_y // 2

            patch = image.crop((start_x, start_y, end_x, end_y)).resize((64, 64))

            if random.randint(1, 10) > 5:
                patch.save(data_path.parent /
                           'classification_patches/negative' / f'{one_annotation_path.name[:-4]}_{idx}.jpg')


if __name__ == '__main__':
    create_positive_patch()
    # create_negative_patch()
