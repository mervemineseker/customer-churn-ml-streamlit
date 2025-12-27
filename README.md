👉 https://churn-ml-shap.streamlit.app/



Interactive Streamlit app for customer churn prediction with model-based explainability.



\# Customer Churn Prediction (Explainable ML + Streamlit)



An end-to-end machine learning project that predicts customer churn probability and explains the key drivers behind each prediction using \*\*linear model contributions\*\*.  

Includes a deployed Streamlit demo with a simple UI and explainability cards.



\## Demo



\- Live Streamlit app: https://churn-ml-shap.streamlit.app/

\- Can also be run locally (instructions below)



\## What this project includes



\- \*\*Model training\*\* with a scikit-learn Pipeline (preprocessing + Logistic Regression)

\- \*\*Explainability\*\* based on \*\*linear model contributions\*\* (feature impact per customer)

\- \*\*Interactive UI\*\* built with Streamlit:

&nbsp; - Churn probability

&nbsp; - Risk level (HIGH / LOW)

&nbsp; - Top drivers shown as cards

\- \*\*Clean repo structure\*\* and reproducible setup via `requirements.txt`

\- \*\*Cloud deployment\*\* on Streamlit Community Cloud



\## Explainability approach



This project uses \*\*logistic regression feature contributions\*\* to explain predictions:



\- Each feature’s contribution is computed as:  

&nbsp; \*\*feature value × model coefficient\*\*

\- Positive impact → increases churn risk  

\- Negative impact → decreases churn risk



This approach is:

\- Stable in cloud environments

\- Fast and deterministic

\- Easy to interpret for business users



\## Dataset



\- Telco Customer Churn dataset (Kaggle / IBM)

\- The raw dataset is \*\*not committed\*\* to the repository

\- The model is trained once and saved as a reusable artifact



\## Project structure



churn-ml/

├── app/

│ └── app.py

├── src/

│ └── train.py

├── notebooks/

├── requirements.txt

└── .gitignore



r

Kodu kopyala



\## How to run locally (Windows)



Create and activate a virtual environment:



```bash

python -m venv .venv

.venv\\Scripts\\activate

Install dependencies:



bash

Kodu kopyala

pip install -r requirements.txt

Train the model (creates models/churn\_pipeline.joblib):



bash

Kodu kopyala

python src/train.py

Start the Streamlit app:



bash

Kodu kopyala

streamlit run app/app.py

Open the Local URL shown in the terminal (usually http://localhost:8501).



Screenshots









Notes

All preprocessing is handled inside the scikit-learn Pipeline



No separate processed files are required



The project can be extended with:



Model comparison (e.g. XGBoost)



Threshold tuning



Monitoring and drift detection

