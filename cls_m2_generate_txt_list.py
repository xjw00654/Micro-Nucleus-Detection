import os
import random


def main():
    data_path = f'data/classification_patches/data_self_annotated_finetuned_PNG'
    positive_patch_path = os.path.join(data_path, 'positive_selected_aug')
    negative_patch_path_1 = os.path.join(data_path, 'PNG_negative_patch_generated_aug+blank')
    negative_patch_path_2 = os.path.join(data_path, 'random_picked_negative_patch_s1_negtive')
    negative_patch_path_Normal = os.path.join(data_path, 'negative_Normal')

    positive_list = [os.path.join(positive_patch_path, elem)
                     for elem in os.listdir(positive_patch_path) if elem.endswith('.png')]
    negative_list = [os.path.join(negative_patch_path_1, elem)
                     for elem in os.listdir(negative_patch_path_1) if elem.endswith('.png')] + \
                    [os.path.join(negative_patch_path_2, elem)
                     for elem in os.listdir(negative_patch_path_2) if elem.endswith('.png')]
    negative_list_Normal = [os.path.join(negative_patch_path_Normal, elem)
                     for elem in os.listdir(negative_patch_path_Normal) if elem.endswith('.jpg')]

    random.shuffle(positive_list)
    random.shuffle(negative_list)

    with open('train_list.txt', 'w') as fp:
        for elem in positive_list[:int(0.9*len(positive_list))]:
            fp.writelines(f'{elem} 1\n')
        for elem in negative_list[:int(0.9*len(negative_list))]:
            fp.writelines(f'{elem} 0\n')

    with open('validation_list.txt', 'w') as fp:
        for elem in positive_list[int(0.9*len(positive_list)):]:
            fp.writelines(f'{elem} 1\n')
        for elem in negative_list[int(0.9*len(negative_list)):]:
            fp.writelines(f'{elem} 0\n')
        for elem in negative_list_Normal:
            fp.writelines(f'{elem} 0\n')


if __name__ == '__main__':
    main()