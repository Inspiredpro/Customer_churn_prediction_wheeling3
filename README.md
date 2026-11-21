# 📉 Customer Churn Risk Predictor

A Streamlit web app that serves a **Logistic Regression** churn model trained on the
[Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d).
Enter a customer's details and get a churn probability, a risk level, the factors behind the score,
and suggested retention actions. Upload a CSV to get a ranked **retention list**.

## What the app does

| Tab | What you get |
|-----|--------------|
| **Single customer** | Risk gauge (0-100%), LOW / MEDIUM / HIGH verdict, a "what if we changed the contract?" comparison, a chart of what drives the score, and rule-based retention suggestions |
| **Score a customer list** | Upload a CSV, get every customer scored and ranked, summary counts, and a downloadable `retention_list.csv` |
| **About** | Plain-language explanation of the model and its limits |

The sidebar lets the business adjust the HIGH / MEDIUM cut-offs (defaults from the notebook: HIGH ≥ 50%, MEDIUM ≥ 30%).

## The model

- **Algorithm:** `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`
- **Target:** `Churned` (1 = left, 0 = stayed)
- **Features (10):**
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
  - Categorical (one-hot encoded with `pd.get_dummies`): `Contract`, `InternetService`, `PaymentMethod`, `PaperlessBilling`, `TechSupport`, `OnlineSecurity`
- **Cleaning:** blank `TotalCharges` values (customers with tenure 0) are removed for training; the app fills them with `tenure × MonthlyCharges` when scoring.

## Project structure

```
churn-predictor/
├── app.py                 # the Streamlit app
├── requirements.txt       # dependencies
├── churn_model.joblib     # YOUR saved model (root or model/ both work)
├── .streamlit/
│   └── config.toml        # colours and server settings
├── .gitignore
└── README.md
```

## 1. Save your model from the notebook

In your notebook, after the comparison cell (Step 16) has trained the models, run:

```python
import joblib, os
os.makedirs("model", exist_ok=True)
joblib.dump(models["Logistic Regression"], "model/churn_model.joblib")
```

Download the file and place it next to `app.py` (or in a `model/` folder).

> `models["Logistic Regression"]` is the full pipeline (scaler + classifier), so no separate scaler file is needed.
> If you saved the scaler and classifier separately, put the scaler at `model/scaler.joblib`. The app detects it automatically.

The app automatically finds any `.joblib`, `.pkl`, `.pickle` or `.sav` file placed next to `app.py` (project root) or inside a `model/` folder, so the file name and location are flexible. A file with "scaler" in its name is treated as the optional scaler.

**Match your scikit-learn version.** In the notebook run `import sklearn; print(sklearn.__version__)` and pin that version in `requirements.txt` (this project is pinned to `scikit-learn==1.6.1`). Mismatched versions can cause load errors or warnings.

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## 3. Deploy with GitHub + Streamlit Community Cloud

1. **Create a GitHub repository** (for example `churn-predictor`) and push this folder, including your model file:
   ```bash
   git init
   git add .
   git commit -m "Churn risk predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/churn-predictor.git
   git push -u origin main
   ```
   The model file is small (a few KB), so it fits in a normal GitHub repo.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**. After a minute or two you get a public URL like `https://<your-app>.streamlit.app`.

Every `git push` to `main` redeploys the app automatically.

## CSV format for batch scoring

Required columns: `tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Contract, InternetService, PaymentMethod, PaperlessBilling, TechSupport, OnlineSecurity`.
The original Telco CSV works as it is (extra columns are kept in the output). Use **Download a template CSV** in the app for an example row.

Allowed category values:

| Column | Values |
|--------|--------|
| Contract | Month-to-month, One year, Two year |
| InternetService | DSL, Fiber optic, No |
| PaymentMethod | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |
| PaperlessBilling | Yes, No |
| TechSupport / OnlineSecurity | Yes, No, No internet service |
| SeniorCitizen | 0 or 1 |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No model file found" | Check the model file is committed to GitHub (`git ls-files`) and sits next to `app.py` or in `model/`, then reboot the app |
| Model fails to load on Streamlit Cloud | Pin `scikit-learn` in `requirements.txt` to the version used in the notebook |
| Predictions look wrong | The model must be trained on scaled, one-hot-encoded features from Steps 8-9. Re-export `models["Logistic Regression"]` |
| Driver chart says "not available" | That chart needs a linear model with `coef_`, such as Logistic Regression |

## Customising

- **Risk cut-offs:** change the slider defaults in the sidebar section of `app.py`.
- **Retention suggestions:** edit `suggested_actions()` in `app.py`.
- **Colours and fonts:** edit `.streamlit/config.toml` and the CSS block at the top of `app.py`.
- **More features:** add them to the lists at the top of `app.py` and retrain the model on the same columns.

## Limitations

- Around a quarter of customers churn, so accuracy alone is misleading. Check recall and precision for churners before trusting the model.
- The model uses 10 features and can only rank risk, not explain individual motives.
- Driver bars show patterns in past data, not proven causes.
