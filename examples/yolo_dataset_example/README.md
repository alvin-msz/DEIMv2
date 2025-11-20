# YOLO Dataset Example

This directory shows the expected structure for a YOLO format dataset.

## Directory Structure

```
yolo_dataset_example/
├── classes.txt          # Class names (one per line)
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image3.txt
        ├── image4.txt
        └── ...
```

## Label Format

Each `.txt` file in the `labels` folder contains annotations for the corresponding image.

Format: `<class_id> <x_center> <y_center> <width> <height>`

All coordinates are normalized to [0, 1].

### Example: image1.txt
```
0 0.5 0.5 0.3 0.4
2 0.2 0.3 0.15 0.2
0 0.8 0.7 0.25 0.35
```

This means:
- Object 1: class 0 (person), center at (50%, 50%), size 30% x 40%
- Object 2: class 2 (car), center at (20%, 30%), size 15% x 20%
- Object 3: class 0 (person), center at (80%, 70%), size 25% x 35%

## Usage

1. Place your images in `images/train/` and `images/val/`
2. Place corresponding labels in `labels/train/` and `labels/val/`
3. Update `classes.txt` with your class names
4. Update your config file:

```yaml
train_dataloader:
  dataset:
    type: YOLODetection
    img_folder: ./examples/yolo_dataset_example/images/train
    label_folder: ./examples/yolo_dataset_example/labels/train
    class_names: ./examples/yolo_dataset_example/classes.txt
```

## Converting from COCO

Use the provided conversion tool:

```bash
python examples/yolo_dataset_example/convert_coco_to_yolo.py \
    --coco_json path/to/annotations.json \
    --img_dir path/to/images \
    --output_dir ./examples/yolo_dataset_example/train
```

## Testing Your Dataset

Test if your dataset loads correctly:

```bash
python examples/yolo_dataset_example/test_yolo_dataset.py \
    --img_folder ./examples/yolo_dataset_example/images/train \
    --class_names ./examples/yolo_dataset_example/classes.txt \
    --num_samples 5 \
    --output_dir ./vis_yolo
```

This will visualize the first 5 samples and save them to `./vis_yolo/`.

