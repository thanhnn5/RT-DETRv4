"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

YOLO to COCO Format Converter

This script converts YOLO format annotations to COCO format.

YOLO format:
- One .txt file per image with the same name as the image
- Each line: <class_id> <x_center> <y_center> <width> <height>
- All coordinates are normalized (0-1)

COCO format:
- Single JSON file with images, annotations, and categories
- Bounding boxes in [x_min, y_min, width, height] format (absolute pixels)

Usage:
    python tools/dataset/yolo_to_coco.py \
        --yolo_root /path/to/yolo/dataset \
        --output_root /path/to/output/dataset \
        --class_names_file /path/to/classes.txt \
        --splits train val
"""

import json
import os
import argparse
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm
import shutil


def _has_exif_orientation(image_path):
    """Return EXIF Orientation value (1-8) or None if no/normal orientation."""
    try:
        with Image.open(image_path) as img:
            ori = img.getexif().get(274, None)
        if ori in (None, 0, 1):
            return None
        return ori
    except Exception:
        return None


def read_oriented_dims(image_path):
    """Read pixel dimensions in DISPLAY orientation (EXIF rotation applied).

    YOLO labels are normalized against what the labeling tool displayed,
    which is the EXIF-rotated view. `cv2.imread` (default, no
    IMREAD_IGNORE_ORIENTATION) applies EXIF orientation in OpenCV >= 4.5.4.
    Pillow's `Image.open(...).size` returns the RAW pixel dims (no auto-
    rotation), so it must NOT be used here.
    """
    arr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if arr is None:
        raise IOError(f"cv2.imread failed: {image_path}")
    h, w = arr.shape[:2]
    return w, h, arr


def save_image_no_exif(image_path, output_path, oriented_array):
    """Copy image to output_path; strip EXIF if orientation is non-trivial.

    For EXIF-rotated images, we write the orientation-applied pixels
    (`oriented_array`, from cv2.imread default) so the saved file has the
    same pixel grid as the recorded COCO width/height — independent of
    whether downstream loads with PIL or cv2.
    """
    if _has_exif_orientation(image_path) is None:
        shutil.copy2(image_path, output_path)
        return
    ext = Path(output_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(str(output_path), oriented_array,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(output_path), oriented_array)


def load_class_names(class_names_file):
    """Load class names from file."""
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def convert_yolo_bbox_to_coco(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bbox format to COCO format.
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x_min, y_min, width, height] in absolute pixels
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert from normalized to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # Convert from center format to top-left format
    x_min = x_center_abs - width_abs / 2
    y_min = y_center_abs - height_abs / 2
    
    return [x_min, y_min, width_abs, height_abs]


def process_split(yolo_images_dir, yolo_labels_dir, output_images_dir, 
                  output_ann_file, class_names, split_name):
    """
    Process a single split (train/val/test).
    
    Args:
        yolo_images_dir: Directory containing YOLO images
        yolo_labels_dir: Directory containing YOLO label txt files
        output_images_dir: Output directory for images
        output_ann_file: Output annotation JSON file path
        class_names: List of class names
        split_name: Name of the split (train/val/test)
    """
    print(f"\nProcessing {split_name} split...")
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_ann_file), exist_ok=True)
    
    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(yolo_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(yolo_images_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")
    
    annotation_id = 1
    image_id = 1
    
    for image_path in tqdm(image_files, desc=f"Converting {split_name}"):
        # Get corresponding label file
        label_path = Path(yolo_labels_dir) / f"{image_path.stem}.txt"
        
        # Read dimensions in display orientation (EXIF rotation applied).
        # YOLO labels are normalized against this frame, NOT the raw pixel
        # grid. Pillow's `.size` returns raw dims, so it is wrong here.
        try:
            img_width, img_height, oriented_array = read_oriented_dims(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Copy image to output directory, stripping EXIF orientation so the
        # saved pixels match the recorded width/height regardless of which
        # loader (PIL/cv2) is used downstream.
        output_image_path = Path(output_images_dir) / image_path.name
        save_image_no_exif(image_path, output_image_path, oriented_array)
        
        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_width,
            "height": img_height
        })
        
        # Process annotations if label file exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Invalid annotation in {label_path}: {line}")
                    continue

                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]

                # Convert YOLO bbox to COCO format
                coco_bbox = convert_yolo_bbox_to_coco(yolo_bbox, img_width, img_height)

                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]

                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    # Save COCO format JSON
    print(f"Saving annotations to {output_ann_file}")
    with open(output_ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Conversion complete!")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert YOLO format dataset to COCO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python tools/dataset/yolo_to_coco.py \\
      --yolo_root /path/to/yolo/dataset \\
      --output_root /path/to/output/dataset \\
      --class_names_file /path/to/classes.txt \\
      --splits train val

YOLO dataset structure expected:
  yolo_root/
  ├── images/
  │   ├── train/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── val/
  │       ├── image1.jpg
  │       └── ...
  └── labels/
      ├── train/
      │   ├── image1.txt
      │   └── ...
      └── val/
          ├── image1.txt
          └── ...

Output COCO dataset structure:
  output_root/
  ├── images/
  │   ├── train/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── val/
  │       ├── image1.jpg
  │       └── ...
  └── annotations/
      ├── instances_train.json
      └── instances_val.json
        """
    )

    parser.add_argument(
        '--yolo_root',
        type=str,
        required=True,
        help='Root directory of YOLO format dataset'
    )

    parser.add_argument(
        '--output_root',
        type=str,
        required=True,
        help='Output directory for COCO format dataset'
    )

    parser.add_argument(
        '--class_names_file',
        type=str,
        required=True,
        help='Path to file containing class names (one per line)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to convert (default: train val)'
    )

    parser.add_argument(
        '--images_subdir',
        type=str,
        default='images',
        help='Subdirectory name for images in YOLO dataset (default: images)'
    )

    parser.add_argument(
        '--labels_subdir',
        type=str,
        default='labels',
        help='Subdirectory name for labels in YOLO dataset (default: labels)'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load class names
    print(f"Loading class names from {args.class_names_file}")
    class_names = load_class_names(args.class_names_file)
    print(f"Found {len(class_names)} classes: {class_names}")

    # Process each split
    for split in args.splits:
        yolo_images_dir = os.path.join(args.yolo_root, args.images_subdir, split)
        yolo_labels_dir = os.path.join(args.yolo_root, args.labels_subdir, split)
        output_images_dir = os.path.join(args.output_root, 'images', split)
        output_ann_file = os.path.join(args.output_root, 'annotations', f'instances_{split}.json')

        # Check if input directories exist
        if not os.path.exists(yolo_images_dir):
            print(f"Warning: Images directory not found: {yolo_images_dir}")
            continue

        if not os.path.exists(yolo_labels_dir):
            print(f"Warning: Labels directory not found: {yolo_labels_dir}")
            continue

        # Process the split
        process_split(
            yolo_images_dir=yolo_images_dir,
            yolo_labels_dir=yolo_labels_dir,
            output_images_dir=output_images_dir,
            output_ann_file=output_ann_file,
            class_names=class_names,
            split_name=split
        )

    print("\n" + "="*50)
    print("All conversions completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
