# PathFL: Multi-alignment Federated Learning for pathology image segmentation

## 1. Project Overview

**PathFL** is a **multi-alignment federated learning framework** for pathology image segmentation. It tackles cross-center heterogeneity through three alignment levels while preserving data privacy:  

- **Image-level:** Collaborative style enhancement boosts data diversity.  
- **Feature-level:** Adaptive alignment ensures consistent representation learning.  
- **Aggregation-level:** Stratified similarity aggregation improves global generalization. 

[*PathFL: Multi-alignment Federated Learning for pathology image segmentation*, Medical Image Analysis, Volume 105, 2025, 103670.](https://www.sciencedirect.com/science/article/pii/S1361841525002178)  
DOI: [10.1016/j.media.2025.103670](https://doi.org/10.1016/j.media.2025.103670)

### Reference

If you find this project helpful, you are welcome to cite our paper:  
```bibtex
@article{Zhang2025PathFL,
  title     = {PathFL: Multi-alignment Federated Learning for pathology image segmentation},
  author    = {Yuan Zhang and Feng Chen and Yaolei Qi and Guanyu Yang and Huazhu Fu},
  journal   = {Medical Image Analysis},
  volume    = {105},
  pages     = {103670},
  year      = {2025},
  issn      = {1361-8415},
  doi       = {10.1016/j.media.2025.103670},
}
```

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

- We use this dataset as an example.  Prepare your medical images and labels according to the format in `Txt/ClusterSeg_fold_1/`.
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
   - A: To simplify the code, we applied simple and general transforms across all centers. Depending on your dataset, you can configure different transforms to achieve better results.
---

If you have any questions or suggestions, feel free to reach out!
yuanzhang_@seu.edu.cn

---

