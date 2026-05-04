import json
import os

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# ANFIS-LLFSR: End-to-End Training\n",
        "This notebook trains the Neuro-Fuzzy Inferencing Based System for Low-Light Face Image Super-Resolution."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!nvidia-smi\n",
        "!pip install scikit-fuzzy lpips pytorch-msssim gdown insightface onnxruntime"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Setup Environment & Mount Drive"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Assuming your code is zipped in Google Drive as 'ANFIS_LLFSR.zip'\n",
        "# !cp /content/drive/MyDrive/ANFIS_LLFSR.zip .\n",
        "# !unzip -q ANFIS_LLFSR.zip -d /content/ANFIS_LLFSR\n",
        "import os\n",
        "os.chdir('/content/ANFIS_LLFSR')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Download Dataset (CelebA)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python scripts/download_data.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Train Classical Pipeline (ANFIS, LCR, Regression)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python app/backend/training/train_full.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Evaluation and Metrics"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python scripts/evaluate.py --n_images 50"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Save Checkpoints to Google Drive"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!cp -r checkpoints /content/drive/MyDrive/ANFIS_LLFSR_checkpoints\n",
        "!cp -r results /content/drive/MyDrive/ANFIS_LLFSR_results\n",
        "print('Saved training checkpoints and results to Google Drive!')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("colab_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("colab_training.ipynb created successfully!")
