# **Regression and Classification on Restaurant Orders**

## **Project Overview**
This project explores various regression and classification techniques applied to a **restaurant orders dataset**. The objective is to model **tipping behavior** and classify **customer satisfaction** and **tip categories** using different machine learning approaches. The implementation utilizes **PyTorch** and includes experiments with **Adam** and **SGD with momentum** optimizers.

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Methods Implemented](#methods-implemented)
5. [Usage](#usage)
6. [Results](#results)
7. [Author](#author)

---

## **Project Structure**
```
|-- Asg2.py                          # Main Python script for regression and classification
|-- Data_Mining_Asg2.pdf              # Report detailing methodology and results
|-- cleaned_restaurant_orders.csv      # Input dataset (not included in repo)
|-- outputs/                          # Directory where generated figures and models are saved
    |-- feature_correlation_heatmap.png
    |-- scatter_plots_features_vs_tip.png
    |-- regression_line_adam.png
    |-- regression_line_sgd.png
    |-- loss_curve_adam_vs_sgd.png
    |-- loss_curve_multi_feature.png
    |-- actual_vs_predicted_multi_feature.png
    |-- polynomial_regression_degree_2.png
    |-- polynomial_regression_degree_4.png
    |-- polynomial_regression_degree_6.png
    |-- roc_curve_binary.png
    |-- confusion_matrix_ova.png
    |-- confusion_matrix_ovo.png
```

---

## **Installation**
### **Dependencies**
This project requires the following Python libraries:
- `torch`
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scikit-learn`

You can install them using:
```bash
pip install torch pandas matplotlib seaborn numpy scikit-learn
```

---

## **Dataset**
The dataset (`cleaned_restaurant_orders.csv`) includes various features related to restaurant bills and tipping behavior:
- `BillAmount`: Total bill amount
- `Spending_per_Guest`: Average spending per guest
- `WaitTime`: Time taken for the order
- `Tip`: Tip amount given by the customer
- `CustomerSatisfaction`: Customer satisfaction level (binary classification target)
- `Is_Weekend`, `Hour`, `NumItems`, `Tip_Percentage`: Additional features used for classification

> **Note:** The dataset file is not included in this repository and needs to be added manually.

---

## **Methods Implemented**
The project consists of **five main parts**:

1. **Single-Feature Linear Regression**  
   - Predicts **tip amount** based on **bill amount**.
   - Models trained with **Adam** and **SGD with momentum**.

2. **Multiple-Feature Linear Regression**  
   - Uses additional features (`NumItems`, `Tip_Percentage`) to improve predictions.
   - Achieves better accuracy compared to single-feature regression.

3. **Polynomial Regression**  
   - Investigates **non-linear relationships** using polynomial transformations.
   - Models trained for degrees **2, 4, and 6** to analyze complexity vs. overfitting.

4. **Binary Classification**  
   - Predicts **customer satisfaction** (`Satisfied` vs. `Unsatisfied`).
   - Models: **Logistic Regression**, **Neural Network (MLP)**.

5. **Multiple-Classes Classification**  
   - Classifies **tip amounts** into three categories: **Low, Medium, High**.
   - Implemented using **One-vs-All (OvA)** and **One-vs-One (OvO)** strategies.

---

## **Usage**
### **Running the Script**
To execute the project, run the following command:
```bash
python Asg2.py
```
This will:
- Load the dataset
- Train regression and classification models
- Generate and save plots in the `outputs/` directory

---

## **Results**
- **Regression Models:**  
  - **Single-feature regression** showed low predictive power (**R² ≈ 0.14**).  
  - **Multi-feature regression** improved accuracy (**R² ≈ 0.69**).  
  - **Polynomial regression (degree 4)** provided the best performance (**R² ≈ 0.83**), but degree **6** overfitted.

- **Classification Models:**  
  - **Binary classification (customer satisfaction)**:  
    - **MLP (84% accuracy)** performed better than **logistic regression (77.7%)**.  
  - **Multi-class classification (tip categories)**:  
    - **OvA (55.3%) and OvO (56.3%)** had moderate performance.  
    - Additional feature engineering may be needed for improved classification.

For a detailed explanation of the results, refer to the **Data_Mining_Asg2.pdf** report.

---

## **Author**
**Benjamin Ramos**  
Panther ID: 5793758  
Course: CAP4770 Data Mining  
Date: February 24, 2025  

---

This README provides an overview of the project, installation steps, dataset details, implemented models, and results. For in-depth analysis, refer to the full **report** (`Data_Mining_Asg2.pdf`). 🚀
