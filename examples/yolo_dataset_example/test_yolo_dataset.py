"""
Test script for YOLO dataset loading
Usage:
    python tools/test_yolo_dataset.py --img_folder dataset/yolo/images/train --class_names dataset/yolo/classes.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from engine.data.dataset import YOLODetection
from engine.data.transforms import Compose


def visualize_sample(dataset, idx, output_dir='./vis_yolo'):
    """Visualize a sample from the dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load item without transforms
    image, target = dataset.load_item(idx)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    
    boxes = target['boxes']
    labels = target['labels']
    
    print(f"\nSample {idx}:")
    print(f"  Image size: {image.size}")
    print(f"  Number of objects: {len(labels)}")
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    colors = [
        'red', 'blue', 'green', 'yellow', 'purple', 
        'orange', 'pink', 'cyan', 'magenta', 'lime'
    ]
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        label_id = label.item()
        
        color = colors[label_id % len(colors)]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        class_name = dataset.class_names[label_id] if label_id < len(dataset.class_names) else f"class_{label_id}"
        text = f"{class_name}"
        
        # Draw text background
        bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), text, fill='white', font=font)
        
        print(f"  Object {i}: class={class_name} (id={label_id}), box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Save visualization
    output_path = output_dir / f"sample_{idx}.jpg"
    image.save(output_path)
    print(f"  Saved visualization to: {output_path}")
    
    return image


def test_dataset(args):
    """Test YOLO dataset loading"""
    print("="*80)
    print("Testing YOLO Dataset Loading")
    print("="*80)
    
    # Create dataset
    print(f"\nLoading dataset from: {args.img_folder}")
    
    dataset = YOLODetection(
        img_folder=args.img_folder,
        label_folder=args.label_folder,
        class_names=args.class_names,
        transforms=None  # No transforms for testing
    )
    
    print(f"\nDataset Info:")
    print(f"  Number of images: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Class names: {dataset.class_names}")
    
    # Test loading a few samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nTesting {num_samples} samples...")
    
    for i in range(num_samples):
        try:
            visualize_sample(dataset, i, args.output_dir)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with DataLoader
    print("\n" + "="*80)
    print("Testing with DataLoader")
    print("="*80)
    
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return images, targets
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    print(f"\nDataLoader created with batch_size={args.batch_size}")
    print("Loading first batch...")
    
    try:
        images, targets = next(iter(loader))
        print(f"  Batch loaded successfully!")
        print(f"  Number of images in batch: {len(images)}")
        print(f"  Image sizes: {[img.size for img in images]}")
        print(f"  Number of objects per image: {[len(t['labels']) for t in targets]}")
    except Exception as e:
        print(f"Error loading batch: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Test YOLO dataset loading')
    parser.add_argument('--img_folder', type=str, required=True,
                        help='Path to images folder')
    parser.add_argument('--label_folder', type=str, default=None,
                        help='Path to labels folder (optional, will auto-infer)')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Path to classes.txt or comma-separated class names')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for DataLoader test')
    parser.add_argument('--output_dir', type=str, default='./vis_yolo',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Parse class_names if it's a comma-separated string
    if args.class_names and ',' in args.class_names:
        args.class_names = [name.strip() for name in args.class_names.split(',')]
    
    test_dataset(args)


if __name__ == '__main__':
    main()

