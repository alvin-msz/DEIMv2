"""
Convert COCO format dataset to YOLO format

Usage:
    python tools/convert_coco_to_yolo.py \
        --coco_json dataset/COCO/annotations/instances_train2017.json \
        --img_dir dataset/COCO/train2017 \
        --output_dir dataset/yolo/train

This will create:
    dataset/yolo/train/images/  (symlinks to original images)
    dataset/yolo/train/labels/  (YOLO format labels)
    dataset/yolo/classes.txt    (class names)
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import os


def coco_to_yolo(coco_json, img_dir, output_dir, use_symlinks=True):
    """
    Convert COCO format to YOLO format
    
    Args:
        coco_json: Path to COCO annotation JSON file
        img_dir: Path to directory containing images
        output_dir: Output directory for YOLO format dataset
        use_symlinks: If True, create symlinks to images instead of copying
    """
    print("="*80)
    print("Converting COCO to YOLO format")
    print("="*80)
    
    # Load COCO annotations
    print(f"\nLoading COCO annotations from: {coco_json}")
    with open(coco_json) as f:
        coco = json.load(f)
    
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping (COCO category_id to 0-indexed class_id)
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {cat['id']: i for i, cat in enumerate(categories)}
    
    print(f"\nCategory mapping:")
    for cat in categories:
        print(f"  {cat['id']:3d} -> {cat_id_to_idx[cat['id']]:3d}: {cat['name']}")
    
    # Create image_id to image info mapping
    img_id_to_info = {img['id']: img for img in coco['images']}
    
    # Group annotations by image_id
    print("\nGrouping annotations by image...")
    img_id_to_anns = {}
    for ann in tqdm(coco['annotations']):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Process each image
    print("\nProcessing images and creating labels...")
    img_dir = Path(img_dir)
    
    skipped_images = 0
    processed_images = 0
    total_objects = 0
    
    for img_id, img_info in tqdm(img_id_to_info.items()):
        file_name = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        
        # Source image path
        src_img_path = img_dir / file_name
        
        # Check if source image exists
        if not src_img_path.exists():
            skipped_images += 1
            continue
        
        # Create image link/copy
        dst_img_path = images_dir / file_name
        if not dst_img_path.exists():
            if use_symlinks:
                # Create relative symlink
                rel_path = os.path.relpath(src_img_path, images_dir)
                os.symlink(rel_path, dst_img_path)
            else:
                # Copy file
                import shutil
                shutil.copy2(src_img_path, dst_img_path)
        
        # Get annotations for this image
        anns = img_id_to_anns.get(img_id, [])
        
        # Write YOLO format labels
        label_file = labels_dir / (Path(file_name).stem + '.txt')
        with open(label_file, 'w') as f:
            for ann in anns:
                # Skip annotations with invalid bbox
                if 'bbox' not in ann or len(ann['bbox']) != 4:
                    continue
                
                x, y, box_w, box_h = ann['bbox']
                
                # Skip invalid boxes
                if box_w <= 0 or box_h <= 0:
                    continue
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x + box_w / 2) / w
                y_center = (y + box_h / 2) / h
                box_w_norm = box_w / w
                box_h_norm = box_h / h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_w_norm = max(0, min(1, box_w_norm))
                box_h_norm = max(0, min(1, box_h_norm))
                
                class_idx = cat_id_to_idx[ann['category_id']]
                
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {box_w_norm:.6f} {box_h_norm:.6f}\n")
                total_objects += 1
        
        processed_images += 1
    
    # Write classes.txt
    classes_file = output_dir.parent / 'classes.txt'
    print(f"\nWriting class names to: {classes_file}")
    with open(classes_file, 'w') as f:
        for cat in categories:
            f.write(cat['name'] + '\n')
    
    # Print summary
    print("\n" + "="*80)
    print("Conversion Summary")
    print("="*80)
    print(f"  Processed images: {processed_images}")
    print(f"  Skipped images: {skipped_images}")
    print(f"  Total objects: {total_objects}")
    print(f"  Output directory: {output_dir}")
    print(f"  Images directory: {images_dir}")
    print(f"  Labels directory: {labels_dir}")
    print(f"  Classes file: {classes_file}")
    print("="*80)
    
    return processed_images, total_objects


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format dataset to YOLO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert training set
  python tools/convert_coco_to_yolo.py \\
      --coco_json dataset/COCO/annotations/instances_train2017.json \\
      --img_dir dataset/COCO/train2017 \\
      --output_dir dataset/yolo/train

  # Convert validation set
  python tools/convert_coco_to_yolo.py \\
      --coco_json dataset/COCO/annotations/instances_val2017.json \\
      --img_dir dataset/COCO/val2017 \\
      --output_dir dataset/yolo/val

  # Copy images instead of creating symlinks
  python tools/convert_coco_to_yolo.py \\
      --coco_json dataset/COCO/annotations/instances_train2017.json \\
      --img_dir dataset/COCO/train2017 \\
      --output_dir dataset/yolo/train \\
      --no-symlinks
        """
    )
    
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO annotation JSON file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--no-symlinks', action='store_true',
                        help='Copy images instead of creating symlinks')
    
    args = parser.parse_args()
    
    coco_to_yolo(
        args.coco_json,
        args.img_dir,
        args.output_dir,
        use_symlinks=not args.no_symlinks
    )
    
    print("\nConversion completed successfully!")
    print("\nYou can now use the YOLO format dataset with:")
    print(f"  img_folder: {Path(args.output_dir) / 'images'}")
    print(f"  label_folder: {Path(args.output_dir) / 'labels'}")
    print(f"  class_names: {Path(args.output_dir).parent / 'classes.txt'}")


if __name__ == '__main__':
    main()

