# Project Setup Guide (10 Commands to Reproducibility)

This guide provides a reproducible, 10-command sequence to set up the ANFIS-LLFSR project on any local machine or server.

## Prerequisites
- Python 3.11+
- (Optional but recommended) NVIDIA GPU with CUDA support for training.

## The 10-Command Setup

Open your terminal and run the following exactly as written:

```bash
# 1. Clone or extract the project repository (assuming extracted in 'LH')
cd LH

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# 4. Install PyTorch (CPU version by default, see PyTorch site for CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install the remaining dependencies
pip install -r app/backend/requirements.txt

# 6. Install frontend dependencies
cd app/frontend
npm install

# 7. Download the Dataset (CelebA)
cd ../../
python scripts/download_data.py

# 8. Start the FastAPI backend server
python app/backend/server.py
# (Leave this running in the terminal)

# 9. In a new terminal, start the React frontend
cd app/frontend
npm run dev
# (Leave this running in the terminal)

# 10. Run a local evaluation (optional)
python scripts/evaluate.py --n_images 10
```

## Running the Training (Colab)
For training, we highly recommend using Google Colab.
1. Upload the `colab_training.ipynb` file to Google Colab.
2. Zip this entire project folder and upload it to Google Drive as `ANFIS_LLFSR.zip`.
3. Select `Runtime -> Change runtime type -> T4 GPU`.
4. Run all cells in the notebook.

## Accessing the Demo
Once Steps 8 and 9 are running, open your browser to:
- **Frontend UI:** `http://localhost:5173`
- **Backend API Docs:** `http://localhost:8001/api/docs`
