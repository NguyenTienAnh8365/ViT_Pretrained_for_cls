# ViT Pretrained for Sea Animal Image Classification

This project demonstrates how to fine-tune a pretrained Vision Transformer (ViT) model for image classification, specifically for classifying sea animal images. The workflow includes data preprocessing, model training, evaluation, and inference using the Hugging Face Transformers library.

---

## 📂 Project Structure

```
ViT_Pretrained_for_cls/
├── ViT_classification.ipynb       # Main notebook: data, training, evaluation, inference
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Ignore unnecessary files
├── .gitattributes                 # Track large files with Git LFS
└── sea_animal_cls_model_pth/     # Directory to hold model weights
    └── model.safetensors          # Model file >100MB
```

---

## 🚀 Features

- Download and preprocess a sea animal image dataset from Kaggle.
- Use Hugging Face's pretrained ViT model for transfer learning.
- Custom PyTorch Dataset wrappers for flexible transforms.
- Training, validation, and evaluation pipeline with metrics.
- Inference on custom images.
- Easily extensible for other image classification tasks.

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/NguyenTienAnh8365/ViT_Pretrained_for_cls.git
    cd ViT_Pretrained_for_cls
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) If using KaggleHub, authenticate your Kaggle account as required.**

---

## 📊 Dataset

- **Source:** [Kaggle - Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)
- **Download:** The notebook uses `kagglehub` to download the dataset automatically.
- **Structure:** Images are organized in subfolders by class.

---

## 📝 Usage

### 1. Open the Notebook

Open `ViT_classification.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

### 2. Run All Cells

The notebook covers:
- Mounting Google Drive (if running on Colab)
- Installing required packages
- Downloading and preprocessing the dataset
- Defining data transforms and loaders
- Setting up the ViT model for classification
- Training and evaluating the model
- Saving and loading the trained model
- Running inference on new images

### 3. Custom Inference

You can test the trained model on your own images by modifying the inference section at the end of the notebook.

---

## 🧩 Key Files

- **ViT_classification.ipynb**  
  Main notebook with all code for data loading, training, evaluation, and inference.

- **requirements.txt**  
  List of required Python packages.

- **.gitignore**  
  Ignore unnecessary files and folders (e.g., checkpoints, cache, data).

---

## 🖼️ Example Results

After training, the notebook will output accuracy metrics and allow you to visualize predictions on test images.

---

## ⚡ Tips

- For local runs, comment out or remove Google Colab-specific code (e.g., `drive.mount`).
- Adjust batch size, learning rate, and epochs in the notebook as needed for your hardware.
- If you want to use a different dataset, update the dataset download and folder paths accordingly.

---

## 📚 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [KaggleHub Documentation](https://github.com/Kaggle/kagglehub)

---

## 👤 Author

- [Nguyen Tien Anh]
- [anhnguyentien8365@gmail.com]

