"""
===========================================
❤️ Heart Disease Prediction Project
===========================================

This project predicts the likelihood of heart disease based on patient health data.
It uses a machine learning pipeline trained with Scikit-learn and provides a web-based
interface built with Streamlit.

-------------------------------------------
📂 Project Structure
-------------------------------------------
Heart_Disease_Project/
│
├── data/                    # Raw and cleaned datasets
├── notebooks/               # Jupyter notebooks for EDA, preprocessing, modeling
├── models/                  # Trained model (.pkl file)
├── ui/                      # Streamlit web app (app.py)
├── requirements.txt         # Python dependencies
├── README.py                # Project documentation (this file)


-------------------------------------------
🚀 Run Locally with Streamlit
-------------------------------------------
1️⃣ Navigate to the UI folder:
    cd ui

2️⃣ Run the Streamlit app:
    streamlit run app.py

3️⃣ Open the local URL (shown in your terminal):
    Example: http://localhost:8501


-------------------------------------------
🧠 Model Details
-------------------------------------------
- Algorithms used: Logistic Regression, Decision Tree, Random Forest, SVM
- Feature Engineering: One-hot encoding, scaling, PCA
- Best model: Tuned Random Forest Classifier
- Exported model: models/final_model.pkl
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, AUC

-------------------------------------------
💡 Example Prediction
-------------------------------------------
The app allows users to enter health parameters such as:
- Age, Blood Pressure, Cholesterol, Heart Rate, etc.
It then predicts whether the user is at risk of heart disease or not.

-------------------------------------------
🧰 Tech Stack
-------------------------------------------
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib
- Ngrok (for public deployment)

-------------------------------------------
📞 Author
-------------------------------------------
Developed by: ELGOSS MOUHCINE
LinkedIn: https://www.linkedin.com/in/mouhcine-elgoss
Email: mohcineelgoss0@gmail.com

===========================================
"""
