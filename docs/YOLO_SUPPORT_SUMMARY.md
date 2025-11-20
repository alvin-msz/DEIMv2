# YOLO Format Support - Implementation Summary

## Overview

YOLO format dataset support has been successfully added to DEIMv2 without changing the existing training paradigm. The implementation is fully compatible with all existing features including data augmentation, evaluation, and distributed training.

## What Was Added

### 1. Core Dataset Class
**File:** `engine/data/dataset/yolo_dataset.py`

A new `YOLODetection` class that:
- Loads images and YOLO format labels (normalized txt files)
- Converts YOLO format (x_center, y_center, width, height) to internal format (x1, y1, x2, y2)
- Provides the same interface as `CocoDetection` and `VOCDetection`
- Supports auto-inference of label folder paths
- Supports multiple ways to specify class names (file path, list, or auto-search)

### 2. Configuration Files

**Dataset Config:** `configs/dataset/yolo_detection.yml`
- Template for YOLO dataset configuration
- Can be included in any model config

**Model Config:** `configs/deimv2/deimv2_hgnetv2_s_yolo.yml`
- Example training config using YOLO dataset
- Based on the COCO config with dataset type changed

### 3. Utility Tools

**Conversion Tool:** `examples/yolo_dataset_example/convert_coco_to_yolo.py`
- Converts COCO format annotations to YOLO format
- Creates symlinks or copies images
- Generates classes.txt file
- Usage:
  ```bash
  python examples/yolo_dataset_example/convert_coco_to_yolo.py \
      --coco_json annotations.json \
      --img_dir images/ \
      --output_dir yolo_dataset/
  ```

**Testing Tool:** `examples/yolo_dataset_example/test_yolo_dataset.py`
- Validates YOLO dataset loading
- Visualizes samples with bounding boxes
- Tests DataLoader compatibility
- Usage:
  ```bash
  python examples/yolo_dataset_example/test_yolo_dataset.py \
      --img_folder dataset/images/train \
      --class_names dataset/classes.txt \
      --num_samples 5
  ```

### 4. Documentation

**Detailed Guide:** `docs/YOLO_FORMAT.md`
- Complete documentation on YOLO format structure
- Configuration examples
- Conversion instructions
- Troubleshooting guide

**Example Dataset:** `examples/yolo_dataset_example/`
- Example directory structure
- Sample classes.txt
- Usage instructions

**Updated README:** `README.md`
- Added YOLO format to supported formats
- Quick start guide for YOLO datasets
- Links to detailed documentation

## YOLO Format Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   └── img2.txt
│   └── val/
│       ├── img3.txt
│       └── img4.txt
└── classes.txt
```

### Label Format
Each line in a label file:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to [0, 1].

## Usage Examples

### Basic Training
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
    -c configs/deimv2/deimv2_hgnetv2_s_yolo.yml \
    --use-amp --seed=0

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    -c configs/deimv2/deimv2_hgnetv2_s_yolo.yml \
    --use-amp --seed=0
```

### Configuration
```yaml
train_dataloader:
  type: DataLoader
  dataset:
    type: YOLODetection
    img_folder: ./dataset/yolo/images/train
    label_folder: ./dataset/yolo/labels/train  # Optional
    class_names: ./dataset/yolo/classes.txt    # Or list of names
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction
```

### Advanced Options

**Auto-infer label folder:**
```yaml
dataset:
  type: YOLODetection
  img_folder: ./dataset/images/train
  # label_folder auto-inferred as ./dataset/labels/train
  class_names: ./dataset/classes.txt
```

**Provide class names as list:**
```yaml
dataset:
  type: YOLODetection
  img_folder: ./dataset/images/train
  class_names: ['person', 'car', 'dog']
```

**Auto-find classes.txt:**
```yaml
dataset:
  type: YOLODetection
  img_folder: ./dataset/images/train
  # Searches for classes.txt in parent directories
```

## Compatibility

The YOLO dataset implementation is fully compatible with:

✅ All data augmentation transforms (RandomFlip, Resize, Mosaic, etc.)
✅ COCO evaluator (for mAP calculation)
✅ All model architectures (HGNetV2, DINOv3, etc.)
✅ Mixed precision training (--use-amp)
✅ Distributed training (torchrun)
✅ EMA (Exponential Moving Average)
✅ Learning rate schedulers
✅ All existing training features

## Implementation Details

### Design Principles
1. **No changes to existing code**: Only additions, no modifications to existing dataset classes
2. **Same interface**: `YOLODetection` follows the same pattern as `CocoDetection` and `VOCDetection`
3. **Internal format conversion**: YOLO format is converted to the same internal format used by COCO
4. **Registered with decorator**: Uses `@register()` for automatic discovery

### Key Features
- **Flexible class specification**: File path, list, or auto-search
- **Auto-inference**: Label folder can be auto-inferred from image folder
- **Robust parsing**: Handles various edge cases (empty files, invalid boxes, etc.)
- **Coordinate clamping**: Ensures boxes stay within image boundaries
- **Format conversion**: YOLO (normalized center) → Internal (absolute xyxy)

### Code Structure
```python
@register()
class YOLODetection(DetDataset):
    def __init__(self, img_folder, label_folder=None, class_names=None, transforms=None)
    def load_item(self, idx) -> (image, target)
    def __getitem__(self, idx) -> (image, target)
    # Plus utility methods and properties for compatibility
```

## Testing

### Unit Test
```bash
python examples/yolo_dataset_example/test_yolo_dataset.py \
    --img_folder dataset/yolo/images/train \
    --class_names dataset/yolo/classes.txt \
    --num_samples 5 \
    --output_dir ./vis_yolo
```

### Integration Test
```bash
# Test with actual training (1 epoch)
CUDA_VISIBLE_DEVICES=0 python train.py \
    -c configs/deimv2/deimv2_hgnetv2_s_yolo.yml \
    --use-amp --seed=0 \
    -u epoches=1
```

## Files Modified/Added

### Added Files
- `engine/data/dataset/yolo_dataset.py` - Core dataset class
- `configs/dataset/yolo_detection.yml` - Dataset config template
- `configs/deimv2/deimv2_hgnetv2_s_yolo.yml` - Model config example
- `examples/yolo_dataset_example/convert_coco_to_yolo.py` - Conversion utility
- `examples/yolo_dataset_example/test_yolo_dataset.py` - Testing utility
- `docs/YOLO_FORMAT.md` - Detailed documentation
- `examples/yolo_dataset_example/` - Example dataset structure
- `YOLO_SUPPORT_SUMMARY.md` - This file

### Modified Files
- `engine/data/dataset/__init__.py` - Added import for `YOLODetection`
- `README.md` - Added YOLO format documentation and examples

## Next Steps

To use YOLO format with your own dataset:

1. **Organize your data** in YOLO format (or convert from COCO)
2. **Create a config file** based on `configs/dataset/yolo_detection.yml`
3. **Update paths** in the config to point to your dataset
4. **Test loading** with `examples/yolo_dataset_example/test_yolo_dataset.py`
5. **Train** using the standard training command

## Support

For issues or questions:
- Check `docs/YOLO_FORMAT.md` for detailed documentation
- Run `examples/yolo_dataset_example/test_yolo_dataset.py` to validate your dataset
- Refer to `examples/yolo_dataset_example/` for structure reference

## Summary

YOLO format support has been seamlessly integrated into DEIMv2 with:
- ✅ Zero changes to existing training pipeline
- ✅ Full compatibility with all features
- ✅ Comprehensive documentation and examples
- ✅ Utility tools for conversion and testing
- ✅ Flexible configuration options

The implementation follows the same patterns as existing dataset classes, ensuring consistency and maintainability.

