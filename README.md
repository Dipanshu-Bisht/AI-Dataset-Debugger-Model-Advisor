# AI Dataset Debugger & Model Advisor

An automated analytics platform that lets you upload any CSV dataset and instantly get
exploratory data analysis, data quality checks, statistical profiling, and ML model
recommendations — all in one interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

🔴 **Live Demo:** https://ai-dataset-debugger-model-advisor--dipanshubisht01.replit.app/

---

## The Problem This Solves

Every data science project starts the same way — load data, check for nulls, check
distributions, try a few models, compare metrics. This takes hours of repetitive work.

This tool automates all of that in seconds. Upload your CSV and get instant answers.

---

## What It Does

- **Data Profiling** — shape, column types, missing values, unique counts, statistical summary
- **Data Quality Checks** — null detection, duplicate records, outlier flagging
- **EDA Visualizations** — distributions, correlations, class balance charts
- **Automated Insights** — auto-generated observations about your dataset
- **ML Model Training** — trains 3 baseline models automatically on your data
- **Model Comparison** — evaluates all models across multiple metrics
- **Best Model Recommendation** — recommends the best model based on performance

---

## Tech Stack

| Area | Tools Used |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |

---

## Project Structure

```
AI-Dataset-Debugger-Model-Advisor/
│
├── analysis/              # Data profiling and quality check logic
├── insights/              # Automated insight generation
├── ui/                    # Streamlit UI components
├── Sample data to test/   # Sample CSV files to try the app
├── app.py                 # Main Streamlit entry point
├── requirements.txt
└── README.md
```

---

## How to Use — Live Demo

1. Go to → https://ai-dataset-debugger-model-advisor--dipanshubisht01.replit.app/
2. Upload any CSV dataset
3. Instantly get:
   - Data profile and quality report
   - EDA visualizations
   - Automated insights
   - ML model comparison and recommendation

No setup needed — runs directly in browser. ✅

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Dipanshu-Bisht/AI-Dataset-Debugger-Model-Advisor.git
cd AI-Dataset-Debugger-Model-Advisor
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Upload a dataset
Open http://localhost:8501 and upload any CSV file.

Sample datasets are available in the `Sample data to test/` folder.

---

## ML Models Compared

The system automatically trains and compares 3 baseline models:

| Model | Type | Best For |
|---|---|---|
| Logistic Regression | Linear | Linearly separable data |
| Decision Tree | Non-linear | Interpretable rules |
| Random Forest | Ensemble | Robust general performance |

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The best model is recommended based on overall F1 Score performance.

---

## Sample Datasets to Test

Sample CSV files are included in the `Sample data to test/` folder.
You can also test with any publicly available dataset from Kaggle.

---

## Key Learnings

- Automating repetitive EDA saves hours of manual work on every new dataset
- A single tool that covers profiling, quality checks, and modeling is more useful than separate scripts
- Modular folder structure (analysis, insights, ui) makes the codebase easy to extend
- Streamlit makes it easy to turn Python scripts into interactive tools without web development knowledge

---

## Author

**Dipanshu Bisht**
- GitHub: https://github.com/Dipanshu-Bisht
- Email: dipanshu.bisht01@gmail.com
- LinkedIn: https://www.linkedin.com/in/dipanshubisht23

---

## License

This project is open source and available under the [MIT License](LICENSE).
