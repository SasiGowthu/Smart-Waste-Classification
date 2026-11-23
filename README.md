# Smart Waste Classification

A beginner-friendly Deep Learning college project using PyTorch, MobileNetV2 transfer learning, and Streamlit.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- PyCharm
- Internet connection for the first installation and pretrained model download

## 2. Dataset
Download the Kaggle dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

Extract it. Copy the folders that represent the waste classes into:

```text
data/garbage/
    battery/
    biological/
    brown-glass/
    cardboard/
    clothes/
    green-glass/
    metal/
    paper/
    plastic/
    shoes/
    trash/
    white-glass/
```

The names above are examples. Use the actual class folders supplied by your downloaded dataset. Each class folder must contain its images. The training script automatically reads folder names as labels.

If the extracted dataset has an extra parent folder, copy the class folders from inside that parent into `data/garbage`.

## 3. Create the PyCharm environment
Open this project folder in PyCharm. Open the Terminal and run:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv` as the project interpreter in PyCharm.

## 4. Train the model
From the project root, run:

```bash
python train.py
```

Optional settings:
```bash
python train.py --epochs 8 --batch_size 16
```

The first run downloads ImageNet pretrained MobileNetV2 weights. Training time depends on your CPU/GPU and dataset size.

Outputs:
- `models/waste_classifier.pth` — best validation model
- `outputs/accuracy.png` — accuracy graph
- `outputs/loss.png` — loss graph
- `outputs/training_history.json` — epoch metrics

## 5. Start the web application
After training finishes:

```bash
python -m streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`. Open it in your browser, upload a JPG/PNG image, and click **Classify Waste**.

## 6. Troubleshooting
- `No module named pip`: run `python -m ensurepip --upgrade`, then `python -m pip install --upgrade pip`.
- `No module named torch`: activate the correct `.venv` and install `requirements.txt` with that interpreter.
- `Dataset folder not found`: confirm that `data/garbage` exists in the project root.
- `Found 0 files`: make sure image files are directly inside class folders (not still buried in an extra nested directory).
- CUDA out-of-memory: reduce batch size with `python train.py --batch_size 8`.
- Slow training: use fewer epochs initially, or train in Google Colab if available.
- Pretrained weights fail to download: check internet access and rerun training.

## 7. Important evaluation note
The script uses a random image-level train/validation split. If the dataset contains duplicate or near-duplicate images, validation accuracy may be optimistic. For a stronger academic evaluation, remove duplicates and use a fixed, documented split. The confidence score is a model score, not a guarantee that the prediction is correct.

## 8. Suggested project report chapters
1. Abstract
2. Introduction
3. Problem statement and objectives
4. Literature review
5. Dataset description and preprocessing
6. Proposed methodology and MobileNetV2 architecture
7. Implementation
8. Results: accuracy, loss, sample predictions, limitations
9. Conclusion and future scope
10. References
