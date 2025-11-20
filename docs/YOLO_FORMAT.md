# YOLO Format Dataset Support

DEIMv2 now supports YOLO format datasets in addition to COCO and VOC formats.

## YOLO Format Structure

Your dataset should be organized as follows:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img3.jpg
│       ├── img4.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   └── val/
│       ├── img3.txt
│       ├── img4.txt
│       └── ...
└── classes.txt
```

### Label Format

Each `.txt` file in the `labels` folder corresponds to an image with the same name in the `images` folder.

Each line in the label file represents one object in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `x_center`: Center X coordinate (normalized to 0-1)
- `y_center`: Center Y coordinate (normalized to 0-1)
- `width`: Box width (normalized to 0-1)
- `height`: Box height (normalized to 0-1)

**Example label file (`img1.txt`):**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
0 0.8 0.7 0.25 0.35
```

### Classes File

The `classes.txt` file contains class names, one per line:

```
person
bicycle
car
motorcycle
...
```

The line number (0-indexed) corresponds to the class ID used in label files.

## Configuration

### Method 1: Use the provided YOLO config template

1. Copy and modify the YOLO dataset config:

```bash
cp configs/dataset/yolo_detection.yml configs/dataset/my_yolo_dataset.yml
```

2. Edit `configs/dataset/my_yolo_dataset.yml`:

```yaml
task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: 80  # Change to your number of classes
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: YOLODetection
    img_folder: /path/to/your/dataset/images/train
    label_folder: /path/to/your/dataset/labels/train  # Optional
    class_names: /path/to/your/dataset/classes.txt
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: YOLODetection
    img_folder: /path/to/your/dataset/images/val
    label_folder: /path/to/your/dataset/labels/val  # Optional
    class_names: /path/to/your/dataset/classes.txt
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
```

3. Create a model config that includes your dataset config:

```bash
cp configs/deimv2/deimv2_hgnetv2_s_yolo.yml configs/deimv2/my_model.yml
```

4. Edit the first line to include your dataset config:

```yaml
__include__: [
  '../dataset/my_yolo_dataset.yml',  # Your dataset config
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml'
]
```

### Method 2: Modify existing COCO config

You can also directly modify an existing config file by changing the dataset type:

```yaml
train_dataloader:
  type: DataLoader
  dataset:
    type: YOLODetection  # Changed from CocoDetection
    img_folder: /path/to/images/train
    class_names: /path/to/classes.txt
    # ... rest of config
```

## Training

Train with YOLO format dataset:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/deimv2/deimv2_hgnetv2_s_yolo.yml \
  --use-amp --seed=0

# Multiple GPUs (if distributed training is available)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  -c configs/deimv2/deimv2_hgnetv2_s_yolo.yml \
  --use-amp --seed=0
```

## Advanced Options

### Auto-infer Label Folder

If your dataset follows the standard YOLO structure (images/labels folders), you can omit `label_folder`:

```yaml
dataset:
  type: YOLODetection
  img_folder: /path/to/dataset/images/train
  # label_folder will be auto-inferred as /path/to/dataset/labels/train
  class_names: /path/to/classes.txt
```

### Provide Class Names as List

Instead of a file path, you can provide class names directly:

```yaml
dataset:
  type: YOLODetection
  img_folder: /path/to/dataset/images/train
  class_names: ['person', 'bicycle', 'car', 'motorcycle']
```

### Auto-find classes.txt

If you don't provide `class_names`, the dataset will search for `classes.txt` in parent directories (up to 3 levels):

```yaml
dataset:
  type: YOLODetection
  img_folder: /path/to/dataset/images/train
  # Will search for classes.txt in:
  # - /path/to/dataset/images/classes.txt
  # - /path/to/dataset/classes.txt
  # - /path/to/classes.txt
```

## Converting from Other Formats

### From COCO to YOLO

```python
import json
from pathlib import Path

def coco_to_yolo(coco_json, output_dir):
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Create output directories
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping
    cat_id_to_idx = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    
    # Process each image
    img_id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    for img in coco['images']:
        img_id = img['id']
        w, h = img['width'], img['height']
        
        # Get annotations for this image
        anns = img_id_to_anns.get(img_id, [])
        
        # Write YOLO format labels
        label_file = labels_dir / (Path(img['file_name']).stem + '.txt')
        with open(label_file, 'w') as f:
            for ann in anns:
                x, y, box_w, box_h = ann['bbox']
                # Convert to YOLO format
                x_center = (x + box_w / 2) / w
                y_center = (y + box_h / 2) / h
                box_w_norm = box_w / w
                box_h_norm = box_h / h
                class_idx = cat_id_to_idx[ann['category_id']]
                f.write(f"{class_idx} {x_center} {y_center} {box_w_norm} {box_h_norm}\n")
    
    # Write classes.txt
    classes_file = Path(output_dir) / 'classes.txt'
    with open(classes_file, 'w') as f:
        for cat in coco['categories']:
            f.write(cat['name'] + '\n')

# Usage
coco_to_yolo('annotations/instances_train2017.json', 'dataset/yolo/train')
```

## Troubleshooting

### Issue: "No images found"
- Check that `img_folder` path is correct
- Ensure images have supported extensions (.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp)

### Issue: "class_names not provided and classes.txt not found"
- Provide `class_names` parameter explicitly
- Or place `classes.txt` in the dataset root directory

### Issue: Empty predictions
- Verify label files exist and are not empty
- Check that class IDs in labels match the number of classes
- Ensure coordinates are normalized to [0, 1]

## Compatibility

The YOLO dataset loader is fully compatible with:
- All existing data augmentation transforms
- COCO evaluator (for mAP calculation)
- All model architectures (HGNetV2, DINOv3, etc.)
- Mixed precision training (--use-amp)
- Distributed training

The dataset internally converts YOLO format to the same format used by COCO datasets, ensuring seamless integration with the existing training pipeline.

