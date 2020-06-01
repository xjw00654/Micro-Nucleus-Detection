from models import *
from utils.utils import *
from utils.datasets import *

import os
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from efficientnet_pytorch import EfficientNet
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def pad_func(patch):
    target_size = np.array([64, 64])
    image = Image.fromarray(patch)

    zero_patch = np.zeros(shape=np.append(target_size, 3), dtype=np.uint8)
    image_size_ratio = target_size / image.size
    image = image.resize([int(e) for e in [elem * min(image_size_ratio) for elem in image.size]])
    pos_idx = (target_size - image.size) // 2
    zero_patch[pos_idx[0]:pos_idx[0] + image.size[0], pos_idx[1]:pos_idx[1] + image.size[1], :] = np.transpose(
        np.array(image), [1, 0, 2])

    return zero_patch


if __name__ == "__main__":
    for case_id in [284, 286, 287, 288]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_folder", type=str, default=f"data/samples/{case_id}/x40", help="path to dataset")
        parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                            help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="checkpoints/hedjitter_after_rm_patch608_yolov3_ckpt_199.pth",
                            help="path to weights file")
        parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
        parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=1216, help="size of each image dimension")
        parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
        opt = parser.parse_args()
        print(opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("output", exist_ok=True)

        # Set up model
        num_results = 0
        net_type = 2
        epoch = 42
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        model_cls = EfficientNet.from_pretrained(f'efficientnet-b{net_type}', num_classes=2).to(device)
        model_cls.load_state_dict(torch.load(f'./checkpoints/PickData_Cls_EffNet-b{net_type}-{epoch}.pkl'))
        model_cls.to(device)
        model_cls.eval()

        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))

        model.eval()  # Set in evaluation mode

        dataloader = DataLoader(
            ImageFolder(opt.image_folder, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        # prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

                name = img_paths[0].split('/')[-1]
                img_40x = Image.open(f'./data/samples/{case_id}/x40/' + name)

                patch_zip, cls_output = [], []

                if detections[0] is None:
                    continue
                for idx, detection in enumerate(detections[0]):
                    x1, y1, x2, y2, _, _, _ = detection
                    x1 = x1.item()
                    y1 = y1.item()
                    x2 = x2.item()
                    y2 = y2.item()

                    patch = np.array(img_40x.crop((x1, y1, x2, y2)))
                    patch = pad_func(patch)
                    patch_tensor = torch.tensor(np.transpose(np.array(patch), (2, 0, 1))).unsqueeze(0)
                    patch_prob = torch.softmax(model_cls(patch_tensor.type(torch.FloatTensor).to(device)), dim=1)
                    patch_output = torch.argmax(patch_prob, dim=1).cpu().numpy()
                    cls_output.append(patch_output[0])
                    if patch_output[0] == 1:
                        Image.fromarray(patch).save(f'/home/jwxie/Desktop/negative_patch_generated/{idx}_{name[:-4]}.png')

                # patch_zip = torch.as_tensor(np.transpose(
                #     np.array(patch_zip, dtype=np.uint8), (0, 3, 1, 2)))
                # cls_output = torch.argmax(model_cls(
                #     patch_zip.type(torch.FloatTensor).to(device)), dim=1).cpu().numpy()


                # o_name = name.replace('.jpg', '')
                # pz = [elem.cpu().numpy() for elem in patch_zip]
                # np.save(f'data/temp/{name}.npy', [pz, cls_output])
                # ##

                # Plot
                idx = np.where(np.array(cls_output) == 1)[0]

                # Draw bounding boxes and labels of detections
                if len(idx) != 0:
                    plt.figure(figsize=(608*2, 608*2))
                    fig, ax = plt.subplots(1)
                    img_40x = np.array(img_40x)
                    ax.imshow(img_40x)

                    # Rescale boxes to original image
                    detections = rescale_boxes(detections[0], 608*2, img_40x.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[idx]:

                        box_w = x2 - x1
                        box_h = y2 - y1
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=(0, 0, 0), facecolor="none")
                        ax.add_patch(bbox)

                    # Save generated image with detections
                    plt.axis("off")
                    plt.gca().xaxis.set_major_locator(NullLocator())
                    plt.gca().yaxis.set_major_locator(NullLocator())
                    filename = img_paths[0].split("/")[-1].split(".")[0]
                    plt.savefig(f"output/two_stage/{filename}.jpg", bbox_inches="tight", pad_inches=0.0, dpi=400)
                    plt.cla()
                    plt.close('all')
                    num_results += 1
                else:
                    filename = img_paths[0].split("/")[-1].split(".")[0]
                    print(f'{filename} skipped for no result, current saved {num_results} positive patch(es).')
