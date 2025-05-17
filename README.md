# Small Cat Recognition Model

This project implements a deep learning pipeline for cat face recognition using PyTorch, EfficientNet, and triplet loss. The model is trained and evaluated on a large-scale cat dataset, supporting efficient similarity search with FAISS.

## Features

- **EfficientNet Backbone**: Uses EfficientNet-B0 for feature extraction.
- **Triplet Loss Training**: Learns discriminative embeddings for cat face identification.
- **Two-Phase Training**: First trains with a frozen backbone, then fine-tunes the entire model.
- **Validation & Visualization**: Evaluates on a validation set and visualizes top predictions.
- **FAISS Integration**: Enables fast similarity search for embeddings.

## Project Structure

- `cat-training (3).ipynb`: Main Jupyter notebook for training, validation, and evaluation.
- `test.py`: Script for evaluating the model on a filtered subset of classes using a CSV file.
- `README.md`: Project overview and instructions.

## Usage

1. **Install Requirements**  
   The notebook installs all required packages automatically:
   ```python
   !pip install timm albumentations faiss-cpu
   ```

2. **Prepare Data**  
   Place the dataset in the expected directory structure (see notebook for details).

3. **Train the Model**  
   Run all cells in `cat-training (3).ipynb` to train and evaluate the model.

4. **Evaluate**  
   Use the notebook or `test.py` to evaluate the model on validation or custom datasets.

## Notes

- The model is designed for large-scale cat face recognition and can be adapted for similar tasks.
- Training and evaluation are optimized for use on GPU-enabled environments (e.g., Kaggle).

## Citation

If you use this code, please cite the original dataset and relevant libraries.

---

**Author:** alfakih  
**License:** MIT
