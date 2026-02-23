"""
Visualization utilities for Face Hallucination System.

This module is SAFE to import in FastAPI and can be used for:
- Visual comparison (Input vs Enhanced vs Super-Resolved)
- Saving figures for reports / papers
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_results(
    input_image: np.ndarray,
    enhanced_image: np.ndarray,
    sr_image: np.ndarray,
    save_path: str | None = None,
    show: bool = False
):
    """
    Visualize input, enhanced, and super-resolved images side by side.

    Args:
        input_image: RGB image [H, W, 3]
        enhanced_image: RGB image [H, W, 3]
        sr_image: RGB image [H, W, 3]
        save_path: Optional path to save the figure
        show: Whether to display the figure interactively
    """

    # Validate inputs
    if input_image is None or enhanced_image is None or sr_image is None:
        raise ValueError("One or more images are None")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_image)
    axes[0].set_title("Input (Low-Light / Low-Res)")
    axes[0].axis("off")

    axes[1].imshow(enhanced_image)
    axes[1].set_title("Enhanced (Zero-DCE)")
    axes[1].axis("off")

    axes[2].imshow(sr_image)
    axes[2].set_title("Super-Resolved (4×)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def visualize_batch(results: list, output_dir: str):
    """
    Visualize and save a batch of pipeline results.

    Args:
        results: List of dictionaries returned by enhance_and_super_resolve()
        output_dir: Directory to save visualizations
    """

    os.makedirs(output_dir, exist_ok=True)

    for idx, result in enumerate(results):
        save_path = os.path.join(output_dir, f"result_{idx}.png")

        visualize_results(
            input_image=result["input_image"],
            enhanced_image=result["enhanced_image"],
            sr_image=result["sr_image"],
            save_path=save_path,
            show=False
        )


def plot_training_curves(history: dict, save_path: str | None = None, show: bool = False):
    """
    Plot training and validation losses from a history dictionary.

    Args:
        history: Dict of metric name -> list of values.
        save_path: Optional path to save the figure.
        show: Whether to display the figure interactively.
    """
    if not history:
        raise ValueError("history is empty")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, values in history.items():
        if values:
            ax.plot(values, label=name)

    ax.set_title("Training Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
