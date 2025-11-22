# Home Credit Default Risk — Complete Version

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## Table of Contents
- [Project Description](#project-description)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Folder](#data-folder)
- [Results and Models](#results-and-models)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description
This project is a complete implementation of the Kaggle competition **Home Credit Default Risk**.  
It predicts the probability of a client defaulting on a loan based on application and credit history data.  
The pipeline covers the full data science process — from data loading to submission file creation —  
focusing on preprocessing, feature engineering, modeling, and evaluation.

The code is **universal** and can run in various environments (Google Colab, Jupyter Notebook, or local machine) without Colab dependencies. Dataset paths can be manually adjusted.

---

## Key Features
- **Data Loading** – Loads all Home Credit datasets (`application_train`, `application_test`, etc.).
- **Data Exploration** – Inspects missing values, data types, duplicates, and anomalies.
- **Preprocessing** – Handles missing values, outliers (IQR capping), normalization, and one-hot encoding.
- **Feature Engineering** – Creates new features (e.g., income-credit ratio) and aggregates related data.
- **Modeling** – Trains multiple models (Logistic Regression, Decision Tree, Random Forest, etc.) with hyperparameter tuning and ensemble (Voting Classifier).
- **Evaluation** – Uses ROC-AUC, ROC curves, and SHAP for interpretability.
- **Submission** – Generates a final `submission.csv` for Kaggle upload.
- **Business Recommendations** – Includes actionable insights for model deployment in real-world use cases.

---

## Dependencies
Install the required Python libraries before running the code:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
````

---

## Installation and Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/FajrinCd/home-credit-default-risk-complete.git
   cd home-credit-default-risk-complete
   ```

2. **Download the dataset**
   Get it from [Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
   and extract all CSV files into a local folder (e.g., `/path/to/home-credit-data/`).

3. **Run the project**
   You can run it in Google Colab, Jupyter Notebook, or any Python environment.

---

## Usage

### 1. Run the Main Code

* Open `home_credit_model.py` (or your chosen filename).
* Input the dataset path when prompted (e.g., `/path/to/home-credit-data/`).
* The code will process data, train models, and generate the final `submission.csv`.

### 2. Outputs

* **Intermediate files:**
  `application_train_merged.csv`, `application_test_merged.csv`,
  `application_train_featured.csv`, `application_test_featured.csv`
* **Final submission:** `submission.csv` (for Kaggle upload)
* **Visualizations:** ROC curve, feature importance, SHAP summary plot

### 3. Example Run (Terminal)

```bash
python home_credit_model.py
```

---

## Project Structure

```
home-credit-default-risk-complete/
├── home_credit_model.py        # Main script (complete pipeline)
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
└── data                        # Folder containing all CSV datasets
```

* `requirements.txt` — Lists all dependencies (see [Dependencies](#dependencies)).
* `data/` (optional) — Contains all Home Credit datasets.
  You can download them from the official [Kaggle Competition Page](https://www.kaggle.com/competitions/home-credit-default-risk/data).

---

## Data Folder

This folder stores all datasets used in the **Home Credit Default Risk** project.
Each file provides information about client applications, credit history, and payment performance.

| File                                                 | Short Description                                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **application_train.csv** / **application_test.csv** | Main loan application data. `TARGET` indicates loan status (0 = repaid, 1 = defaulted). |
| **bureau.csv**                                       | Clients’ credit history from other financial institutions.                              |
| **bureau_balance.csv**                               | Monthly data about previous credits from the `bureau` dataset.                          |
| **previous_application.csv**                         | Clients’ previous loan applications at Home Credit.                                     |
| **POS_CASH_BALANCE.csv**                             | Monthly records of previous point-of-sale or cash loans.                                |
| **credit_card_balance.csv**                          | Monthly balance data for previous credit cards.                                         |
| **installments_payment.csv**                         | Payment history for previous loans (both made and missed).                              |

> Tip: Store all datasets inside the `data/` folder for easier access, or adjust file paths in your code if stored elsewhere.

---

## Results and Models

* **Best Model:** Random Forest or Voting Classifier (ROC-AUC ≈ 0.75+, depending on tuning).
* **Evaluation:** ROC curves visualize performance; SHAP highlights key predictive features like `EXT_SOURCE_2`.
* **Recommendations:** Deploy as part of a credit risk monitoring dashboard using external score features for segmentation.

---

## Contributing

Contributions are welcome!

* Fork this repository and create a new branch for your feature or fix.
* Submit a pull request with a clear description of your changes.
* Report bugs or suggestions via GitHub Issues.

---

## License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project for educational or non-commercial purposes.

---

For questions or feedback, open an issue on GitHub or contact the maintainer: **[dgartup@gmail.com](mailto:dgartup@gmail.com)**
Happy coding! 🚀
