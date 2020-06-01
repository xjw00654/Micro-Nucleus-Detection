# Micro-Nucleus Detection 

## Method 
1. Using the [YOLO-v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) for detection of all cells.
2. Using the [EfficientNet-b2](https://github.com/lukemelas/EfficientNet-PyTorch) for classification of cell types .

p.s. Using this separated 2-step detection rather than the end-2-end RCNN workflow for better
finetuning model and modifying dataset.

## Result
![284_3_17_1_1.jpg](output/two_stage/284_3_17_1_1.jpg)
![286_11_14_0_0.jpg](output/two_stage/286_11_14_0_0.jpg)  
![287_14_4_0_0.jpg](output/two_stage/287_14_4_0_0.jpg)
![288_1_10_1_0.jpg](output/two_stage/288_1_10_1_0.jpg)

