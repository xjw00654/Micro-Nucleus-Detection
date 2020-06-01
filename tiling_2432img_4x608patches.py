import os
import cv2
import shutil
import linecache


def main():
    data_root = 'data/MicroNuclei/annotated_data/images'
    label_root = 'data/MicroNuclei/annotated_data/label'

    save_data_path = './data/MicroNuclei/patch_data_608/images'
    save_label_path = './data/MicroNuclei/patch_data_608/labels'

    images_list = sorted(os.listdir(data_root))
    labels_list = sorted(os.listdir(label_root))

    for img_name, label_name in zip(images_list, labels_list):
        img = cv2.imread(os.path.join(data_root, img_name)).copy()
        label = linecache.getlines(os.path.join(label_root, label_name))
        true_w, true_h, _ = img.shape

        for r in range(4):
            for c in range(4):
                patch = img[c * 608: (c + 1) * 608, r * 608: (r + 1) * 608]

                patch_label = []
                patch_bound = ((r * 608, (r + 1) * 608), (c * 608, (c + 1) * 608))
                for one_label in label:
                    cls, x, y, w, h = one_label.strip().split(' ')
                    x = int(float(x) * true_w)
                    y = int(float(y) * true_h)
                    w = int(float(w) * true_w)
                    h = int(float(h) * true_h)

                    center = (x, y)
                    # tl = (x - int(w // 2), y - int(h // 2))
                    # br = (x + int(w // 2), y + int(h // 2))

                    if patch_bound[0][0] < center[0] < patch_bound[0][1] and \
                            patch_bound[1][0] < center[1] < patch_bound[1][1]:
                        center_xy = [center[0] - r * 608, center[1] - c * 608]
                        patch_label.append(['0', center_xy[0] / 608., center_xy[1] / 608., w / 608., h / 608.])
                if len(patch_label) != 0:
                    patch_path = f'{save_data_path}/{r}_{c}_{img_name}'
                    label_path = f'{save_label_path}/{r}_{c}_{label_name}'
                    cv2.imwrite(patch_path, patch)
                    with open(label_path, 'w') as fp:
                        for line in patch_label:
                            fp.writelines(" ".join(map(str, line)) + '\n')


def rm_patches():
    rm_data_list = [
        '0_1_285_27_14', '0_1_285_2_7', '0_2_284_19_13', '0_2_284_20_19', '1_0_284_5_9', '1_0_284_17_13',
        '1_0_288_29_10', '1_2_284_31_3', '1_2_288_23_5', '1_3_284_3_16', '2_0_284_13_19', '3_1_284_24_17',
        '3_1_284_30_9', '3_2_284_12_1', '3_2_287_15_4']

    save_data_path = './data/MicroNuclei/patch_data_608/images'
    save_label_path = './data/MicroNuclei/patch_data_608/labels'

    for elem in rm_data_list:
        if os.path.exists(os.path.join(save_data_path, elem+'.jpg')):
            shutil.move(os.path.join(save_data_path, elem+'.jpg'),
                        os.path.join('./data/MicroNuclei/patch_data_608/images_rm', elem+'.jpg'))
            shutil.move(os.path.join(save_label_path, elem+'.txt'),
                        os.path.join('./data/MicroNuclei/patch_data_608/labels_rm', elem+'.txt'))


def create_txt_file():
    patch_path = f'./data/MicroNuclei/patch_data_608/images/'
    label_path = f'./data/MicroNuclei/patch_data_608/labels/'
    data_list = os.listdir('./data/MicroNuclei/patch_data_608/images')

    with open('./data/MicroNuclei/patch_data_608/train.txt', 'w') as fp_tr:
        with open('./data/MicroNuclei/patch_data_608/valid.txt', 'w') as fp_va:
            for data in data_list:
                line = os.path.join(patch_path, data) + '\n'
                if not os.path.exists(os.path.join(label_path, data.replace('jpg', 'txt'))):
                    raise FileNotFoundError(f'Not found label file: {data}')
                fp_tr.writelines(line)
                fp_va.writelines(line)


if __name__ == '__main__':
    main()
    rm_patches()
    create_txt_file()
