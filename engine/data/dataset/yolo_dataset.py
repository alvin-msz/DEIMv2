"""
DEIMv2: YOLO Format Dataset Support
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
"""

import os
import torch
import torch.utils.data
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

Image.MAX_IMAGE_PIXELS = None

__all__ = ['YOLODetection']


@register()
class YOLODetection(DetDataset):
    """
    YOLO format dataset loader.
    
    YOLO format structure:
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── val/
    │       ├── img3.jpg
    │       └── img4.jpg
    └── labels/
        ├── train/
        │   ├── img1.txt
        │   └── img2.txt
        └── val/
            ├── img3.txt
            └── img4.txt
    
    Each label file contains lines in format:
    <class_id> <x_center> <y_center> <width> <height>
    where all coordinates are normalized to [0, 1]
    
    Args:
        img_folder: Path to images folder (e.g., 'dataset/images/train')
        label_folder: Path to labels folder (e.g., 'dataset/labels/train'). 
                      If None, will auto-infer by replacing 'images' with 'labels'
        class_names: List of class names or path to classes.txt file
        transforms: Transforms to apply
    """
    __inject__ = ['transforms']
    
    def __init__(
        self, 
        img_folder: str,
        label_folder: Optional[str] = None,
        class_names: Optional[list] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__()
        
        self.img_folder = Path(img_folder)
        
        # Auto-infer label folder if not provided
        if label_folder is None:
            label_folder = str(self.img_folder).replace('images', 'labels')
        self.label_folder = Path(label_folder)
        
        # Load class names
        if class_names is None:
            # Try to find classes.txt in parent directories
            classes_file = self._find_classes_file()
            if classes_file and classes_file.exists():
                with open(classes_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                raise ValueError(
                    "class_names not provided and classes.txt not found. "
                    "Please provide class_names as a list or path to classes.txt"
                )
        elif isinstance(class_names, str):
            # class_names is a path to file
            with open(class_names, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # class_names is a list
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        self._transforms = transforms
        
        # Collect all image files
        self.img_files = self._collect_image_files()
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {self.img_folder}")
        
        print(f"Loaded {len(self.img_files)} images from {self.img_folder}")
    
    def _find_classes_file(self):
        """Try to find classes.txt in parent directories"""
        current = self.img_folder
        for _ in range(3):  # Search up to 3 levels
            current = current.parent
            classes_file = current / 'classes.txt'
            if classes_file.exists():
                return classes_file
        return None
    
    def _collect_image_files(self):
        """Collect all image files from img_folder"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        img_files = []
        
        for ext in img_extensions:
            img_files.extend(self.img_folder.glob(f'*{ext}'))
            img_files.extend(self.img_folder.glob(f'*{ext.upper()}'))
        
        # Sort for reproducibility
        img_files = sorted(img_files)
        return img_files
    
    def _get_label_path(self, img_path: Path):
        """Get corresponding label file path for an image"""
        label_path = self.label_folder / (img_path.stem + '.txt')
        return label_path
    
    def __len__(self):
        return len(self.img_files)
    
    def load_item(self, idx):
        """Load image and annotations"""
        img_path = self.img_files[idx]
        label_path = self._get_label_path(img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Initialize target
        target = {
            'image_id': torch.tensor([idx]),
            'idx': torch.tensor([idx]),
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
            'orig_size': torch.tensor([w, h])
        }
        
        # Load annotations if label file exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                # Parse YOLO format: class_id x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
                
                # Convert from normalized YOLO format to absolute xyxy format
                x1 = (x_center - box_w / 2) * w
                y1 = (y_center - box_h / 2) * h
                x2 = (x_center + box_w / 2) * w
                y2 = (y_center + box_h / 2) * h
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                target['boxes'].append([x1, y1, x2, y2])
                target['labels'].append(class_id)
                target['area'].append((x2 - x1) * (y2 - y1))
                target['iscrowd'].append(0)
        
        # Convert to tensors
        if len(target['boxes']) > 0:
            boxes = torch.tensor(target['boxes'], dtype=torch.float32)
            target['boxes'] = convert_to_tv_tensor(
                boxes, 'boxes', box_format='xyxy', spatial_size=[h, w]
            )
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['area'] = torch.tensor(target['area'], dtype=torch.float32)
            target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)
        else:
            # Empty annotations
            target['boxes'] = convert_to_tv_tensor(
                torch.zeros((0, 4), dtype=torch.float32), 
                'boxes', box_format='xyxy', spatial_size=[h, w]
            )
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
        
        return image, target
    
    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target
    
    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n'
        s += f' label_folder: {self.label_folder}\n'
        s += f' num_classes: {self.num_classes}\n'
        s += f' num_images: {len(self.img_files)}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
    
    @property
    def categories(self):
        """Return categories in COCO format for compatibility"""
        return [{'id': i, 'name': name} for i, name in enumerate(self.class_names)]
    
    @property
    def category2name(self):
        return {i: name for i, name in enumerate(self.class_names)}
    
    @property
    def category2label(self):
        return {i: i for i in range(self.num_classes)}
    
    @property
    def label2category(self):
        return {i: i for i in range(self.num_classes)}

