# 🫀 Heart Disease Prediction App

This repository contains a machine learning application built with **Streamlit** to predict the likelihood of heart disease based on clinical data.

The app leverages data science tools and machine learning algorithms to classify patients into two categories: **with heart disease** or **without heart disease**, based on 13 medical attributes.

---

## 📂 Project Structure

```
heart-disease-prediction/
│
├── streamlit_app.py                 # Streamlit application
├── Heart_Disease_Dataset.csv        # Dataset used
├── Heart_Disease_Prediction.ipynb   # Jupyter Notebook used to train the Models
├── knn_model.pkl                    # Trained KNN model
├── svc_model.pkl                    # Trained Support Vector Classifier model
├── dt_model.pkl                     # Trained Decision Tree model
├── rf_model.pkl                     # Trained Random Forest model
├── requirements.txt                 # List of dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Features

- 📊 Interactive data exploration
- 🧹 Data cleaning and preprocessing
- 📉 Visual insights and EDA (Correlation heatmap, histograms, pairplots)
- 🤖 ML Model training (KNN, SVC, Decision Tree, Random Forest)
- 🏆 Model comparison to select the best performing model
- 🧠 Predict heart disease using trained models
- 💾 Models are serialized using `pickle` for reuse

---

## 📈 Machine Learning Models Used

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVC)** - with various kernels
3. **Decision Tree Classifier**
4. **Random Forest Classifier**

Each model was trained on a processed dataset and evaluated for accuracy using test sets. The best performing versions were saved as `.pkl` files.

---

## 📋 How to Use

### 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MoteneJan/Heart_Disease_Prediction.git
   cd Heart_Disease_Prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### ▶️ Run the App

```bash
streamlit run streamlit_app.py
```

---

## 🛠 App Functionality (Streamlit Pages)

| Page        | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| **Home**    | Overview of the dataset, statistics, and visuals.                           |
| **EDA**     | Plots: Correlation matrix, histograms, and pairplots.                       |
| **Models**  | Compares all trained models based on accuracy.                              |
| **Predict** | User inputs clinical data to predict heart disease using best model(s).     |

---

## 📊 Dataset Info

- **Source:** UCI Heart Disease Dataset
- **Rows:** 303
- **Features:** 13 + 1 target
- **Target:** Binary classification (0 = No Disease, 1 = Disease)

---

## 💾 Model Saving

All final models were saved using `pickle`:
```python
pickle.dump(model, open("model_name.pkl", "wb"))
```

---

## 📌 Future Improvements

- Add model explanations using SHAP or LIME
- Include precision/recall/F1 score visualizations
- Deploy the model via Heroku or Streamlit Cloud
- Collect feedback from medical professionals for improvements

---

## 🙌 Acknowledgements

- Inspired by medical prediction problems from UCI Machine Learning Repository.
- Built using `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `streamlit`.

---

## 👨‍💻 Author

**Jan Motene**  
_Data Science & Software Development Enthusiast_  

---

Let me know if you want me to help write the `streamlit_app.py` or the `requirements.txt` too!
