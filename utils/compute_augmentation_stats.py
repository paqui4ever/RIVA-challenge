"""
Compute batch statistics of augmented datasets.

This script analyzes the effects of data augmentation transforms on the dataset,
computing statistics like:
- Average/max/min number of remaining targets after augmentation
- Bounding box size distributions (width, height, area)
- Class distribution changes
- Comparison between original and augmented data
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.secrets import get_sam3_processor
from data.dataset import BethesdaDataset
from data.transforms import (
    get_train_transforms_DETR,
    get_train_transforms_RCNN,
    get_valid_transforms,
    get_valid_transforms_DETR,
)


class AugmentationStatsCollector:
    """Collects and computes statistics for augmented datasets."""

    def __init__(self, image_size: int = 1008):
        self.image_size = image_size
        self.reset()

    def reset(self):
        """Reset all collected statistics."""
        self.num_targets_per_image: List[int] = []
        self.bbox_widths: List[float] = []
        self.bbox_heights: List[float] = []
        self.bbox_areas: List[float] = []
        self.labels: List[int] = []
        self.images_with_zero_targets: int = 0
        self.total_images: int = 0

    def collect_from_batch(self, targets: List[Dict]):
        """Collect statistics from a batch of targets."""
        for target in targets:
            boxes = target["boxes"]
            labels = target["labels"]

            num_targets = len(boxes)
            self.num_targets_per_image.append(num_targets)
            self.total_images += 1

            if num_targets == 0:
                self.images_with_zero_targets += 1
                continue

            # Compute bbox statistics
            for box in boxes:
                x_min, y_min, x_max, y_max = box.tolist()
                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                self.bbox_widths.append(width)
                self.bbox_heights.append(height)
                self.bbox_areas.append(area)

            # Collect labels
            self.labels.extend(labels.tolist())

    def compute_statistics(self) -> Dict:
        """Compute summary statistics from collected data."""
        stats = {}

        # Target count statistics
        num_targets = np.array(self.num_targets_per_image)
        stats["targets"] = {
            "total_images": self.total_images,
            "total_targets": int(num_targets.sum()),
            "mean": float(num_targets.mean()) if len(num_targets) > 0 else 0,
            "std": float(num_targets.std()) if len(num_targets) > 0 else 0,
            "min": int(num_targets.min()) if len(num_targets) > 0 else 0,
            "max": int(num_targets.max()) if len(num_targets) > 0 else 0,
            "median": float(np.median(num_targets)) if len(num_targets) > 0 else 0,
            "images_with_zero_targets": self.images_with_zero_targets,
            "zero_target_ratio": self.images_with_zero_targets / self.total_images
            if self.total_images > 0
            else 0,
        }

        # Bounding box size statistics
        if len(self.bbox_widths) > 0:
            widths = np.array(self.bbox_widths)
            heights = np.array(self.bbox_heights)
            areas = np.array(self.bbox_areas)

            stats["bbox_width"] = {
                "mean": float(widths.mean()),
                "std": float(widths.std()),
                "min": float(widths.min()),
                "max": float(widths.max()),
                "median": float(np.median(widths)),
                "percentile_5": float(np.percentile(widths, 5)),
                "percentile_95": float(np.percentile(widths, 95)),
            }

            stats["bbox_height"] = {
                "mean": float(heights.mean()),
                "std": float(heights.std()),
                "min": float(heights.min()),
                "max": float(heights.max()),
                "median": float(np.median(heights)),
                "percentile_5": float(np.percentile(heights, 5)),
                "percentile_95": float(np.percentile(heights, 95)),
            }

            stats["bbox_area"] = {
                "mean": float(areas.mean()),
                "std": float(areas.std()),
                "min": float(areas.min()),
                "max": float(areas.max()),
                "median": float(np.median(areas)),
                "percentile_5": float(np.percentile(areas, 5)),
                "percentile_95": float(np.percentile(areas, 95)),
            }

            # Aspect ratio
            aspect_ratios = widths / np.maximum(heights, 1e-6)
            stats["aspect_ratio"] = {
                "mean": float(aspect_ratios.mean()),
                "std": float(aspect_ratios.std()),
                "min": float(aspect_ratios.min()),
                "max": float(aspect_ratios.max()),
            }

            # Relative sizes (normalized by image size)
            rel_widths = widths / self.image_size
            rel_heights = heights / self.image_size
            rel_areas = areas / (self.image_size ** 2)

            stats["relative_sizes"] = {
                "width_mean_pct": float(rel_widths.mean() * 100),
                "height_mean_pct": float(rel_heights.mean() * 100),
                "area_mean_pct": float(rel_areas.mean() * 100),
            }
        else:
            stats["bbox_width"] = None
            stats["bbox_height"] = None
            stats["bbox_area"] = None
            stats["aspect_ratio"] = None
            stats["relative_sizes"] = None

        # Class distribution
        if len(self.labels) > 0:
            labels = np.array(self.labels)
            unique, counts = np.unique(labels, return_counts=True)
            stats["class_distribution"] = {
                int(label): int(count) for label, count in zip(unique, counts)
            }
            stats["num_classes"] = len(unique)
        else:
            stats["class_distribution"] = {}
            stats["num_classes"] = 0

        return stats


def collate_fn(batch):
    """Custom collate function for object detection."""
    return tuple(zip(*batch))


def run_analysis(
    dataset: BethesdaDataset,
    collector: AugmentationStatsCollector,
    num_passes: int = 1,
    batch_size: int = 4,
    num_workers: int = 0,
    desc: str = "Analyzing",
) -> Dict:
    """
    Run analysis on a dataset.

    Args:
        dataset: The dataset to analyze
        collector: Statistics collector
        num_passes: Number of passes through the dataset (useful for stochastic augmentations)
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        desc: Description for progress bar

    Returns:
        Dictionary of computed statistics
    """
    collector.reset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    for pass_idx in range(num_passes):
        pass_desc = f"{desc} (pass {pass_idx + 1}/{num_passes})" if num_passes > 1 else desc
        for images, targets in tqdm(dataloader, desc=pass_desc):
            collector.collect_from_batch(targets)

    return collector.compute_statistics()


def print_statistics(stats: Dict, title: str = "Statistics"):
    """Pretty print statistics."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    # Target statistics
    t = stats["targets"]
    print("\n📊 Target Count Statistics:")
    print(f"  Total images processed: {t['total_images']}")
    print(f"  Total targets: {t['total_targets']}")
    print(f"  Targets per image: {t['mean']:.2f} ± {t['std']:.2f}")
    print(f"  Range: [{t['min']}, {t['max']}] (median: {t['median']:.1f})")
    print(f"  Images with zero targets: {t['images_with_zero_targets']} ({t['zero_target_ratio']*100:.2f}%)")

    # Bounding box statistics
    if stats["bbox_width"] is not None:
        print("\n📐 Bounding Box Size Statistics:")
        bw = stats["bbox_width"]
        bh = stats["bbox_height"]
        ba = stats["bbox_area"]
        print(f"  Width:  {bw['mean']:.1f} ± {bw['std']:.1f} px  (range: [{bw['min']:.1f}, {bw['max']:.1f}])")
        print(f"  Height: {bh['mean']:.1f} ± {bh['std']:.1f} px  (range: [{bh['min']:.1f}, {bh['max']:.1f}])")
        print(f"  Area:   {ba['mean']:.1f} ± {ba['std']:.1f} px²")
        print(f"  Width [5th-95th percentile]:  [{bw['percentile_5']:.1f}, {bw['percentile_95']:.1f}]")
        print(f"  Height [5th-95th percentile]: [{bh['percentile_5']:.1f}, {bh['percentile_95']:.1f}]")

        ar = stats["aspect_ratio"]
        print(f"\n  Aspect ratio (W/H): {ar['mean']:.2f} ± {ar['std']:.2f} (range: [{ar['min']:.2f}, {ar['max']:.2f}])")

        rs = stats["relative_sizes"]
        print(f"\n  Relative to image size ({int(stats.get('image_size', 1008))}x{int(stats.get('image_size', 1008))}):")
        print(f"    Mean width:  {rs['width_mean_pct']:.2f}%")
        print(f"    Mean height: {rs['height_mean_pct']:.2f}%")
        print(f"    Mean area:   {rs['area_mean_pct']:.4f}%")

    # Class distribution
    if stats["class_distribution"]:
        print(f"\n🏷️  Class Distribution ({stats['num_classes']} classes):")
        total = sum(stats["class_distribution"].values())
        for cls, count in sorted(stats["class_distribution"].items()):
            pct = count / total * 100
            print(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")


def compare_statistics(original: Dict, augmented: Dict):
    """Compare original and augmented statistics."""
    print("\n" + "=" * 60)
    print(" Comparison: Original vs Augmented")
    print("=" * 60)

    # Target retention
    orig_t = original["targets"]
    aug_t = augmented["targets"]

    if orig_t["total_images"] > 0 and aug_t["total_images"] > 0:
        # Normalize by number of passes
        orig_per_image = orig_t["total_targets"] / orig_t["total_images"]
        aug_per_image = aug_t["total_targets"] / aug_t["total_images"]
        retention = aug_per_image / orig_per_image * 100 if orig_per_image > 0 else 0

        print(f"\n📊 Target Retention:")
        print(f"  Original avg targets/image: {orig_per_image:.2f}")
        print(f"  Augmented avg targets/image: {aug_per_image:.2f}")
        print(f"  Retention rate: {retention:.1f}%")
        print(f"  Zero-target image increase: {aug_t['zero_target_ratio']*100 - orig_t['zero_target_ratio']*100:+.2f}%")

    # Bounding box size changes
    if original["bbox_area"] is not None and augmented["bbox_area"] is not None:
        print(f"\n📐 Bounding Box Size Changes:")
        for metric in ["bbox_width", "bbox_height", "bbox_area"]:
            orig = original[metric]["mean"]
            aug = augmented[metric]["mean"]
            change = (aug - orig) / orig * 100 if orig > 0 else 0
            unit = "px²" if "area" in metric else "px"
            name = metric.replace("bbox_", "").capitalize()
            print(f"  {name}: {orig:.1f} -> {aug:.1f} {unit} ({change:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute batch statistics of augmented datasets"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV annotation file",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the images directory",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["detr", "rcnn", "both"],
        default="both",
        help="Which transform to analyze (default: both)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=3,
        help="Number of passes through dataset for augmented statistics (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loading (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1008,
        help="Image size for transforms (default: 1008)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for statistics (optional)",
    )

    args = parser.parse_args()

    # Verify paths exist
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.images):
        print(f"Error: Images directory not found: {args.images}")
        sys.exit(1)

    print(f"📁 CSV file: {args.csv}")
    print(f"📁 Images directory: {args.images}")
    print(f"🔄 Number of passes: {args.num_passes}")
    print(f"📏 Image size: {args.image_size}")

    collector = AugmentationStatsCollector(image_size=args.image_size)
    all_results = {}

    # Analyze without augmentation (baseline)
    print("\n" + "-" * 60)
    print("Analyzing baseline (no augmentation)...")
    baseline_dataset = BethesdaDataset(
        csv_file=args.csv,
        root_dir=args.images,
        transforms=get_valid_transforms(),
    )
    baseline_stats = run_analysis(
        baseline_dataset,
        collector,
        num_passes=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc="Baseline",
    )
    baseline_stats["image_size"] = args.image_size
    print_statistics(baseline_stats, "Baseline (No Augmentation)")
    all_results["baseline"] = baseline_stats

    # Analyze RCNN transforms
    if args.transform in ["rcnn", "both"]:
        print("\n" + "-" * 60)
        print("Analyzing RCNN transforms...")
        rcnn_dataset = BethesdaDataset(
            csv_file=args.csv,
            root_dir=args.images,
            transforms=get_train_transforms_RCNN(size=args.image_size),
        )
        rcnn_stats = run_analysis(
            rcnn_dataset,
            collector,
            num_passes=args.num_passes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc="RCNN augmentation",
        )
        rcnn_stats["image_size"] = args.image_size
        print_statistics(rcnn_stats, "RCNN Transforms (Augmented)")
        compare_statistics(baseline_stats, rcnn_stats)
        all_results["rcnn"] = rcnn_stats

    # Analyze DETR transforms
    if args.transform in ["detr", "both"]:
        print("\n" + "-" * 60)
        print("Analyzing DETR transforms...")
        try:
            processor = get_sam3_processor()
            detr_dataset = BethesdaDataset(
                csv_file=args.csv,
                root_dir=args.images,
                transforms=get_train_transforms_DETR(processor, size=args.image_size),
            )
            detr_stats = run_analysis(
                detr_dataset,
                collector,
                num_passes=args.num_passes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="DETR augmentation",
            )
            detr_stats["image_size"] = args.image_size
            print_statistics(detr_stats, "DETR Transforms (Augmented)")
            compare_statistics(baseline_stats, detr_stats)
            all_results["detr"] = detr_stats
        except Exception as e:
            print(f"⚠️  Could not analyze DETR transforms: {e}")
            print("   (Sam3Processor may not be available)")

    # Save results to CSV if requested
    if args.output:
        rows = []
        for transform_name, stats in all_results.items():
            row = {"transform": transform_name}

            # Flatten nested stats
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"{key}_{subkey}"] = subvalue
                else:
                    row[key] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\n📄 Statistics saved to: {args.output}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
