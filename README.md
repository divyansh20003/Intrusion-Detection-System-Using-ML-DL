# Intrusion-Detection-System-Using-ML-DL
📌 Project Overview
As cyber threats continue to rise, this project focuses on developing a robust Intrusion Detection System (IDS) that leverages Machine Learning (ML) and Deep Learning (DL) techniques. Trained on the NSL-KDD dataset, this system aims to effectively detect and classify malicious network activities in both binary (normal vs. attack) and multi-class (DoS, Probe, U2R, R2L) scenarios.

🛠️ Tools & Technologies
Programming Language: Python

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

Deep Learning: TensorFlow

Development Environment: Jupyter Notebook

📁 Dataset
Source: NSL-KDD Dataset (an enhanced version of KDD Cup 1999)

Preprocessing:

Duplicate removal

Irrelevant feature elimination

Encoding of categorical features

RobustScaler used for normalization

Labels converted for binary and multi-class classification

⚙️ Methodology
1. Data Preprocessing
Load and clean training/test datasets

Encode categorical values

Scale features

Categorize attack types

2. Dimensionality Reduction
Applied PCA (Principal Component Analysis) to improve model performance and reduce training time

3. Model Training & Evaluation
Models Used:

Support Vector Machine (SVM)

Random Forest

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Naive Bayes

XGBoost

Deep Neural Network (TensorFlow)

Evaluation:

Binary Classification: Normal vs Attack

Multi-Class Classification: DoS, R2L, Probe, U2R

📊 Results
Model	Accuracy (Binary)	Accuracy (Multi-Class)
SVM	~83%	~80%
Random Forest	~99%	~98%
XGBoost	~99.2%	~98.5%
Deep Neural Net	~98%	~96%

✅ XGBoost and Random Forest achieved the highest accuracy.

🧠 Conclusion
This project demonstrates the effectiveness of ML and DL in network intrusion detection. Incorporating PCA improved performance, and models like XGBoost and DNN achieved high detection accuracy. The system provides a promising foundation for future deployment in real-time enterprise environments.
