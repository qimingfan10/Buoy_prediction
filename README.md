# Buoy Prediction

## 📥 Environment configuration

### Clone repository and install dependencies
```bash
git clone https://github.com/qimingfan10/Buoy_prediction.git
cd Buoy_prediction

pip install -r requirements.txt
```

## 📂 Data preparation
1. Place the CSV file of the site to be predicted in the 'data/predict' directory
2. Data file format requirements:

**Sample file (data.csv)**:
```csv
lon	lat	depth	temperature	salinity
160.9519	89.2999	6	-1.6776	30.6667
160.9519	89.2999	7	-1.6775	30.6665
160.9519	89.2999	8	-1.6776	30.6689
160.9519	89.2999	9	-1.6775	30.6688
```

## 🚀 Execute prediction
```bash
python predict_mamba.py
```

## 📌 Note
1. Ensure that CSV files are separated by **tabs** (not commas)
2. The data file must contain complete header information
3. It is recommended to run in Python 3.8+environment

## 📁 Directory structure
```
Buoy_prediction/
├── data/
│   └── predict/       
├── predict_mamba.py          
├── requirements.txt   
└── README.md          
```

## ⚙️ Dependent environment
- Python 3.8+
- PyTorch >= 1.12
- Pandas >= 1.4
- NumPy >= 1.22
- For other dependencies, please refer to requirements.txt
