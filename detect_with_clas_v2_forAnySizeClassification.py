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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples/285/x20", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_299.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    num_results = 0
    net_type = 4
    epoch = 10
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model_cls = EfficientNet.from_pretrained(f'efficientnet-b{net_type}', num_classes=2).to(device)
    model_cls.load_state_dict(torch.load(f'./checkpoints/Cls_EffNet-b{net_type}-{epoch}.pkl'))
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

            img_40x = Image.open('./data/samples/285/x40/' + img_paths[0].split('/')[-1].replace('.jpg', '.jpg'))

            patch_zip = []

            if detections[0] is None:
                continue
            for detection in detections[0]:
                x1, y1, x2, y2, _, _, _ = detection
                x1 = x1.item()
                y1 = y1.item()
                x2 = x2.item()
                y2 = y2.item()

                img_40x.crop((2 * x1, 2 * y1, 2 * x2, 2 * y2)).save()

            patch_zip = torch.as_tensor(np.transpose(
                np.array(patch_zip, dtype=np.uint8), (0, 3, 1, 2)))
            cls_output = torch.argmax(model_cls(
                patch_zip.type(torch.FloatTensor).to(device)), dim=1).cpu().numpy()

            # Plot
            idx = np.where(cls_output == 1)[0]

            # Draw bounding boxes and labels of detections
            if len(idx) != 0:
                plt.figure(figsize=(608, 608))
                fig, ax = plt.subplots(1)
                img_40x = np.array(img_40x)
                ax.imshow(img_40x)

                # Rescale boxes to original image
                detections = rescale_boxes(detections[0], 608, img_40x.shape[:2])
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
                plt.savefig(f"output/{filename}.jpg", bbox_inches="tight", pad_inches=0.0, dpi=300)
                plt.cla()
                plt.close('all')
                num_results += 1
            else:
                filename = img_paths[0].split("/")[-1].split(".")[0]
                print(f'{filename} skipped for no result, current saved {num_results} positive patch(es).')
