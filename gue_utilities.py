#!/usr/bin/env python
"""
Utility functions for working with GUE data in ModernDNABERT.

This script provides helper functions for working with the GUE benchmark data,
including data loading, preprocessing, and analysis.
"""

import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class GUEDataProcessor:
    """
    Processor for GUE benchmark data.
    """

    def __init__(self, gue_path: str):
        """
        Initialize the GUE data processor.

        Args:
            gue_path: Path to the GUE benchmark data
        """
        self.gue_path = Path(gue_path)
        self.task_categories = {
            "EMP": self.gue_path / "GUE" / "EMP",
            "prom": self.gue_path / "GUE" / "prom",
            "splice": self.gue_path / "GUE" / "splice",
            "virus": self.gue_path / "GUE" / "virus",
            "mouse": self.gue_path / "GUE" / "mouse",
            "tf": self.gue_path / "GUE" / "tf",
        }

        # Task metadata
        self.task_metadata = {
            "EMP": {
                "description": "Epigenetic marker prediction",
                "species": "Human",
                "task_type": "Classification",
                "max_length": 128
            },
            "prom": {
                "description": "Promoter identification",
                "species": "Human",
                "task_type": "Classification",
                "max_length": {
                    "core": 20,
                    "300": 70
                }
            },
            "splice": {
                "description": "Splice site prediction",
                "species": "Human",
                "task_type": "Classification",
                "max_length": 80
            },
            "virus": {
                "description": "Virus classification",
                "species": "Virus",
                "task_type": "Classification",
                "max_length": 256
            },
            "mouse": {
                "description": "Mouse-specific tasks",
                "species": "Mouse",
                "task_type": "Classification",
                "max_length": 30
            },
            "tf": {
                "description": "Transcription factor binding",
                "species": "Human",
                "task_type": "Classification",
                "max_length": 30
            }
        }

    def get_all_task_paths(self) -> Dict[str, Path]:
        """
        Get paths to all tasks in GUE.

        Returns:
            Dictionary mapping task names to their paths
        """
        task_paths = {}

        for category, category_path in self.task_categories.items():
            if not category_path.exists():
                logger.warning(f"Task category path does not exist: {category_path}")
                continue

            for task_dir in category_path.iterdir():
                if task_dir.is_dir():
                    task_name = f"{category}_{task_dir.name}"
                    task_paths[task_name] = task_dir

        return task_paths

    def get_task_info(self) -> pd.DataFrame:
        """
        Get information about all tasks in GUE.

        Returns:
            DataFrame with task information
        """
        task_paths = self.get_all_task_paths()

        task_info = []
        for task_name, task_path in task_paths.items():
            category = task_name.split("_")[0]

            # Training samples
            train_path = task_path / "train.csv"
            dev_path = task_path / "dev.csv"
            test_path = task_path / "test.csv"

            train_samples = 0
            dev_samples = 0
            test_samples = 0
            avg_seq_length = 0
            num_labels = 0

            if train_path.exists():
                train_df = pd.read_csv(train_path)
                train_samples = len(train_df)
                avg_seq_length += train_df["sequence"].str.len().mean() * train_samples
                if "label" in train_df.columns:
                    num_labels = len(train_df["label"].unique())

            if dev_path.exists():
                dev_df = pd.read_csv(dev_path)
                dev_samples = len(dev_df)
                avg_seq_length += dev_df["sequence"].str.len().mean() * dev_samples

            if test_path.exists():
                test_df = pd.read_csv(test_path)
                test_samples = len(test_df)
                avg_seq_length += test_df["sequence"].str.len().mean() * test_samples

            total_samples = train_samples + dev_samples + test_samples

            if total_samples > 0:
                avg_seq_length /= total_samples

            # Get metadata
            metadata = self.task_metadata.get(category, {})
            description = metadata.get("description", "")
            species = metadata.get("species", "")
            task_type = metadata.get("task_type", "")

            # Get max_length based on task name
            max_length = metadata.get("max_length", 512)
            if isinstance(max_length, dict):
                # For prom tasks, determine if it's core or 300
                if "core" in task_name:
                    max_length = max_length.get("core", 512)
                elif "300" in task_name:
                    max_length = max_length.get("300", 512)
                else:
                    max_length = 512

            task_info.append({
                "Task": task_name,
                "Category": category,
                "Description": description,
                "Species": species,
                "Task Type": task_type,
                "Train Samples": train_samples,
                "Dev Samples": dev_samples,
                "Test Samples": test_samples,
                "Total Samples": total_samples,
                "Avg Sequence Length": round(avg_seq_length),
                "Num Labels": num_labels,
                "Max Length": max_length
            })

        return pd.DataFrame(task_info)

    def analyze_sequence_characteristics(self, task_name: str) -> Dict:
        """
        Analyze sequence characteristics for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with sequence characteristics
        """
        task_paths = self.get_all_task_paths()

        if task_name not in task_paths:
            raise ValueError(f"Task {task_name} not found")

        task_path = task_paths[task_name]

        train_path = task_path / "train.csv"
        dev_path = task_path / "dev.csv"
        test_path = task_path / "test.csv"

        dfs = []
        for path in [train_path, dev_path, test_path]:
            if path.exists():
                dfs.append(pd.read_csv(path))

        if not dfs:
            raise ValueError(f"No data found for task {task_name}")

        df = pd.concat(dfs)

        # Analyze sequence lengths
        seq_lengths = df["sequence"].str.len()

        # Analyze nucleotide composition
        nucleotide_counts = {
            "A": 0,
            "T": 0,
            "G": 0,
            "C": 0,
            "other": 0
        }

        for seq in df["sequence"]:
            for c in seq:
                if c in nucleotide_counts:
                    nucleotide_counts[c] += 1
                else:
                    nucleotide_counts["other"] += 1

        total_nucleotides = sum(nucleotide_counts.values())
        nucleotide_composition = {
            k: v / total_nucleotides for k, v in nucleotide_counts.items()
        }

        # Label distribution
        if "label" in df.columns:
            label_distribution = df["label"].value_counts().to_dict()
        else:
            label_distribution = {}

        return {
            "task_name": task_name,
            "num_sequences": len(df),
            "sequence_length": {
                "min": seq_lengths.min(),
                "max": seq_lengths.max(),
                "mean": seq_lengths.mean(),
                "median": seq_lengths.median()
            },
            "nucleotide_composition": nucleotide_composition,
            "label_distribution": label_distribution
        }

    def visualize_task_distribution(self, output_path: Optional[str] = None):
        """
        Visualize the distribution of tasks in GUE.

        Args:
            output_path: Path to save the visualization
        """
        task_info = self.get_task_info()

        # Plot distribution of tasks by category
        plt.figure(figsize=(12, 6))

        # Count tasks by category
        category_counts = task_info["Category"].value_counts()

        # Create bar plot
        ax = sns.barplot(x=category_counts.index, y=category_counts.values)

        # Add labels
        for i, v in enumerate(category_counts.values):
            ax.text(i, v + 0.1, str(v), ha="center")

        plt.title("Distribution of GUE Tasks by Category")
        plt.xlabel("Category")
        plt.ylabel("Number of Tasks")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()

    def visualize_sequence_lengths(self, output_path: Optional[str] = None):
        """
        Visualize the distribution of sequence lengths across all tasks.

        Args:
            output_path: Path to save the visualization
        """
        task_info = self.get_task_info()

        plt.figure(figsize=(15, 8))

        # Create bar plot for average sequence length
        ax = sns.barplot(x="Task", y="Avg Sequence Length", data=task_info)

        # Rotate x-axis labels
        plt.xticks(rotation=90)

        plt.title("Average Sequence Length by Task")
        plt.xlabel("Task")
        plt.ylabel("Average Sequence Length (bp)")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()

    def prepare_task_for_alibi(self, task_name: str, output_dir: str):
        """
        Prepare a task for training with ALiBi attention.

        This function adjusts the sequence length to be compatible with ALiBi attention
        and performs any necessary preprocessing.

        Args:
            task_name: Name of the task
            output_dir: Directory to save the prepared data
        """
        task_paths = self.get_all_task_paths()

        if task_name not in task_paths:
            raise ValueError(f"Task {task_name} not found")

        task_path = task_paths[task_name]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get task info
        task_info = self.get_task_info()
        task_row = task_info[task_info["Task"] == task_name].iloc[0]

        # Get suggested max length for this task
        max_length = task_row["Max Length"]

        logger.info(f"Preparing task {task_name} for ALiBi attention")
        logger.info(f"Original average sequence length: {task_row['Avg Sequence Length']}")
        logger.info(f"Suggested max length for tokenized input: {max_length}")

        # Process each split
        for split in ["train", "dev", "test"]:
            split_path = task_path / f"{split}.csv"

            if split_path.exists():
                df = pd.read_csv(split_path)

                # Save with the same structure
                output_path = os.path.join(output_dir, f"{split}.csv")
                df.to_csv(output_path, index=False)

                logger.info(f"Saved {split} split to {output_path}")

        # Save task metadata
        metadata = {
            "task_name": task_name,
            "category": task_row["Category"],
            "description": task_row["Description"],
            "species": task_row["Species"],
            "task_type": task_row["Task Type"],
            "avg_sequence_length": task_row["Avg Sequence Length"],
            "max_length": max_length,
            "num_labels": task_row["Num Labels"]
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved task metadata to {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GUE Data Processing Utilities")

    parser.add_argument("--gue_path", type=str, required=True,
                        help="Path to the GUE benchmark data")
    parser.add_argument("--action", type=str, required=True,
                        choices=["info", "analyze", "visualize", "prepare"],
                        help="Action to perform")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name for analysis or preparation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for visualizations or prepared data")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Initialize processor
    processor = GUEDataProcessor(args.gue_path)

    # Perform requested action
    if args.action == "info":
        task_info = processor.get_task_info()
        print(task_info.to_string(index=False))

        if args.output:
            task_info.to_csv(args.output, index=False)
            logger.info(f"Saved task info to {args.output}")

    elif args.action == "analyze":
        if not args.task:
            raise ValueError("Task name must be provided for analysis")

        analysis = processor.analyze_sequence_characteristics(args.task)
        print(json.dumps(analysis, indent=4))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=4)
            logger.info(f"Saved analysis to {args.output}")

    elif args.action == "visualize":
        if not args.output:
            logger.warning("No output path provided. Visualizations will be displayed but not saved.")

        processor.visualize_task_distribution(
            output_path=os.path.join(args.output, "task_distribution.png") if args.output else None
        )

        processor.visualize_sequence_lengths(
            output_path=os.path.join(args.output, "sequence_lengths.png") if args.output else None
        )

    elif args.action == "prepare":
        if not args.task:
            raise ValueError("Task name must be provided for preparation")

        if not args.output:
            raise ValueError("Output directory must be provided for preparation")

        processor.prepare_task_for_alibi(args.task, args.output)