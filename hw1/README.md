```makefile
hw1
├── model_base
│   ├── main.py
│   ├── ...
│   └── model.py
├── dataset
│   ├── test.csv
│   └── train.csv
├── main.py
├── ...		
└── README.md 
```

# PM2.5 Prediction (HW1)

This project implements a linear regression model from scratch using NumPy to predict PM2.5 levels.

## How to Run
1. Make sure you have `pandas` and `numpy` installed.
2. Place `train.csv` and `test.csv` in the `./dataset/` directory.
3. Run the main script:
   ```bash
   python main.py