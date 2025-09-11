# Federated Learning Medical Image Segmentation Project

## 1. Project Overview

This project implements a federated learning system for medical image segmentation. It enables multiple hospitals (or centers) to collaboratively train a segmentation model **without sharing raw data**, thus protecting privacy and improving model generalization.

**Audience:** This guide is written for beginners with no coding background, to help you quickly understand and use the project.

---

## 2. Project Structure & Main Files

- `Train.py`: Main script for training and testing, supports all centers.
- `Train_Utils.py`: Utility functions for training, prediction, logging, etc.
- `Params_fusion.py`: Federated parameter fusion algorithm.
- `Network.py`: Neural network definition (e.g., UNet).
- `Loss.py`: Loss function definitions.
- `MyDataloader.py`: Data loader.
- `Utils.py`: Other helper functions.
- `requirement.txt`: List of required Python packages.
- `Readme.md`: This guide.
- Folders like `Txt/`, `Results/`, `Weights/` store data, results, and model weights.

---

## 3. Installation & Running

### 0. (Recommended) Create a New Conda Environment

It is recommended to use [conda](https://docs.conda.io/) to create an isolated environment for this project. In your terminal, run:

```bash
conda create -n PathFL python=3.12 -y
conda activate PathFL
```

### 1. Install Dependencies

Make sure you have Python 3.7 or above. In your terminal, navigate to the project folder and run:

```bash
cd /path/to/your/project
pip install -r requirement.txt
```

### 2. Prepare Data

- Prepare your medical images and labels according to the format in `Txt/ClusterSeg_fold_1/`.
- Each center (0, 1, 2) needs its own training and testing image list (`.txt` files).

### 3. Train the Model **in Parallel** (Examples for Each Center)

**For Center 0:**
```bash
python Train.py --center_id 0
```

**For Center 1:**
```bash
python Train.py --center_id 1
```

**For Center 2:**
```bash
python Train.py --center_id 2 
```

- `--center_id` should be 0, 1, or 2 for each center.
- You can run these commands on different machines for each center at the same time.

### 4. Test the Model

After training, testing will run automatically. Results are saved in the `Results/` folder.

---

## 4. Main Parameter Descriptions

- `--center_id`: Center index, 0/1/2.
- `--base_dir`: Root directory for your data and results.
- `--n_channels`: Number of input image channels (default: 3).
- `--n_classes`: Number of output classes (default: 1).
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate.
- `--n_epochs`: Total number of training epochs.
- `--if_onlytest`: Only run testing (True/False).

**See the argparse section at the top of Train.py for more parameters.**

---

## 5. FAQ

1. **Q: Can I run this without a GPU?**  
   - A: Yes, the code will automatically use the CPU if no GPU is available, but it will be slower.  

2. **Q: How do I add a new center?**  
   - A: Prepare the `.txt` files for the new center and run with a new `center_id`.  

3. **Q: The training seems stuck?**  
   - A: It may be waiting for model parameters from other centers. Make sure all centers are running in parallel.  

4. **Q: Where are logs and results saved?**  
   - A: Logs are saved in the `Log/` folder, and segmentation results are saved in the `Results/` folder.  

5. **Q: Why are the results slightly different from those reported in the paper?**  
   - A: To simplify the code, we used a minimal set of transforms across all centers. Depending on the characteristics of your dataset, you can configure different transforms to achieve better results.

---



If you have any questions or suggestions, feel free to reach out!

---

**Enjoy using the project! If you have any questions, please contact the developer.**
