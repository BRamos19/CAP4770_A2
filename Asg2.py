# =============================================================================
# Data Mining Project – Regression and Classification on Restaurant Orders
#
# This script performs the following:
#   1. Data loading and preprocessing (including outlier removal, feature 
#      engineering, and normalization).
#   2. Single-feature linear regression (comparing Adam vs. SGD with momentum).
#   3. Multi-feature linear regression.
#   4. Polynomial regression with multiple degrees.
#
# Dependencies: torch, pandas, matplotlib, seaborn, numpy, scikit-learn
# =============================================================================

# --------------------------
# Import Libraries
# --------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# Setup Output Directory
# --------------------------
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_current_figure(filename):
    """
    Save the current matplotlib figure to the outputs folder and then display it.
    """
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.show()

# --------------------------
# Define File Path and Load Dataset
# --------------------------
FILE_PATH = "C:\\Users\\bramo\\Python Projects\\CAP4770-A2\\cleaned_restaurant_orders.csv"

# Load dataset and verify required columns
df = pd.read_csv(FILE_PATH)
expected_columns = ["BillAmount", "Spending_per_Guest", "WaitTime", "Tip"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"ERROR: Missing columns in dataset: {missing_columns}")
else:
    print("Dataset loaded successfully. All expected columns are present.")

sns.set_style("whitegrid")

# --------------------------
# Utility Functions and Data Visualization
# --------------------------
def load_dataset(file_path):
    """Load CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

# Plot feature correlation heatmap
df_numeric = df.select_dtypes(include=['number'])
if not df_numeric.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    save_current_figure("feature_correlation_heatmap.png")
else:
    print("No numeric features found for correlation matrix.")

# Create scatter plots for 'Tip' vs. selected features
selected_features = ["BillAmount", "Spending_per_Guest", "WaitTime"]
if "Tip" in df.columns:
    sns.pairplot(df, x_vars=selected_features, y_vars=["Tip"], kind="scatter")
    plt.suptitle("Scatter Plots of Features vs. Tip", y=1.02)
    save_current_figure("scatter_plots_features_vs_tip.png")
else:
    print("'Tip' column not found. Skipping scatter plot.")

def preprocess_data(df, feature_cols, target_col, normalize=True):
    """
    Extract features and target, convert to tensors, and optionally normalize.
    
    Returns:
      - X_tensor: Feature tensor.
      - y_tensor: Target tensor.
      - X_mean, X_std: Mean and standard deviation of features (if normalized).
      - y_mean, y_std: Mean and standard deviation of target (if normalized).
    """
    if target_col not in df.columns:
        raise KeyError(f"ERROR: Column '{target_col}' not found in dataset!")

    X = df[feature_cols].values
    y = df[[target_col]].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    if normalize:
        X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0)
        X_std[X_std == 0] = 1.0

        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        if y_std == 0:
            y_std = 1.0

        X_tensor = (X_tensor - X_mean) / X_std
        y_tensor = (y_tensor - y_mean) / y_std

        return X_tensor, y_tensor, X_mean, X_std, y_mean, y_std
    else:
        return X_tensor, y_tensor, None, None, None, None

def plot_regression_line(X_orig, y_orig, X_tensor, model, X_mean, X_std, y_mean, y_std,
                         title="Regression Plot", xlabel="Feature", ylabel="Target", filename="regression_plot.png"):
    """
    Plot the original data and the regression line.
    Converts normalized predictions back to the original scale.
    """
    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).detach().flatten()
    
    y_pred_orig = predicted * y_std + y_mean
    sorted_indices = X_orig.argsort()
    X_sorted = X_orig[sorted_indices]
    y_pred_sorted = y_pred_orig.numpy()[sorted_indices]

    plt.figure(figsize=(7, 5))
    plt.scatter(X_orig, y_orig, alpha=0.5, label="Data Points")
    plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label="Regression Line")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    save_current_figure(filename)

def polynomial_features(X, degree):
    """
    Create polynomial features (without cross-terms) for each feature up to the given degree.
    """
    poly_terms = [X ** d for d in range(1, degree + 1)]
    return torch.cat(poly_terms, dim=1)

class LinearRegressionModel(nn.Module):
    """
    Simple Linear Regression Model for one or multiple features.
    """
    def __init__(self, input_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def train_regression_model(model, X, y, learning_rate=0.001, epochs=1000, print_every=100,
                           weight_decay=0.0, early_stopping_patience=None, optimizer=None):
    """
    Train a regression model using Mean Squared Error loss.
    
    Returns:
      - Trained model.
      - Loss history (list).
    """
    criterion = nn.MSELoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if early_stopping_patience is not None:
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[Early Stopping] Epoch {epoch}, Loss: {loss.item():.4f}")
                break
        
        if (epoch + 1) % print_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    return model, loss_history

# ----- Single-Feature Regression (BillAmount -> Tip) -----
df = load_dataset(FILE_PATH)
feature_cols = ["BillAmount"]
target_col = "Tip"
if target_col not in df.columns:
    raise KeyError(f"ERROR: '{target_col}' column is missing from dataset!")
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(df, feature_cols, target_col, normalize=True)

model_adam = LinearRegressionModel(input_dim=1)
model_sgd = LinearRegressionModel(input_dim=1)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)

model_adam, loss_adam = train_regression_model(model_adam, X_tensor, y_tensor, learning_rate=0.001,
                                               epochs=1000, print_every=100, weight_decay=0.0,
                                               early_stopping_patience=None, optimizer=optimizer_adam)
model_sgd, loss_sgd = train_regression_model(model_sgd, X_tensor, y_tensor, learning_rate=0.001,
                                             epochs=1000, print_every=100, weight_decay=0.0,
                                             early_stopping_patience=None, optimizer=optimizer_sgd)

with torch.no_grad():
    y_pred_adam = model_adam(X_tensor)
    mse_loss_adam = nn.MSELoss()(y_pred_adam, y_tensor)
    y_pred_sgd = model_sgd(X_tensor)
    mse_loss_sgd = nn.MSELoss()(y_pred_sgd, y_tensor)
print(f"Final MSE (Adam, normalized): {mse_loss_adam.item():.4f}")
print(f"Final MSE (SGD, normalized): {mse_loss_sgd.item():.4f}")

y_pred_actual_adam = y_pred_adam * y_std + y_mean
y_pred_actual_sgd = y_pred_sgd * y_std + y_mean
y_actual = y_tensor * y_std + y_mean
ss_total = torch.sum((y_actual - torch.mean(y_actual)) ** 2)
ss_residual_adam = torch.sum((y_actual - y_pred_actual_adam) ** 2)
r2_adam = 1 - (ss_residual_adam / ss_total)
ss_residual_sgd = torch.sum((y_actual - y_pred_actual_sgd) ** 2)
r2_sgd = 1 - (ss_residual_sgd / ss_total)
print(f"R² Score (Adam, unnormalized): {r2_adam.item():.4f}")
print(f"R² Score (SGD, unnormalized): {r2_sgd.item():.4f}")

plt.figure(figsize=(7, 5))
plt.plot(loss_adam, label="Adam", color="blue")
plt.plot(loss_sgd, label="SGD with Momentum", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve: Adam vs. SGD")
plt.legend()
save_current_figure("loss_curve_adam_vs_sgd.png")

X_orig = df[feature_cols[0]].values
y_orig = df[target_col].values
plot_regression_line(X_orig, y_orig, X_tensor, model_adam, X_mean, X_std, y_mean, y_std,
                     title="Single-Feature Regression (Adam)",
                     xlabel="Bill Amount ($)", ylabel="Tip ($)",
                     filename="regression_line_adam.png")
plot_regression_line(X_orig, y_orig, X_tensor, model_sgd, X_mean, X_std, y_mean, y_std,
                     title="Single-Feature Regression (SGD with Momentum)",
                     xlabel="Bill Amount ($)", ylabel="Tip ($)",
                     filename="regression_line_sgd.png")

# ----- Multi-Feature Regression -----
df = load_dataset(FILE_PATH)
features = ["BillAmount", "NumItems", "Tip_Percentage"]
target_col = "Tip"
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(df, features, target_col, normalize=True)
print("Feature Means:", X_mean.numpy())
print("Feature Standard Deviations:", X_std.numpy())
print("Target Mean:", y_mean.numpy())
print("Target Standard Deviation:", y_std.numpy())
model_multi = LinearRegressionModel(input_dim=len(features))
print("Model type before training:", type(model_multi))
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=0)
model_multi, loss_multi = train_regression_model(model_multi, X_train, y_train, learning_rate=0.0003,
                                                  epochs=2000, print_every=200, weight_decay=0.001)
print("Model type after training:", type(model_multi))
with torch.no_grad():
    y_pred_train = model_multi(X_train)
    y_pred_val = model_multi(X_val)
mse_loss_train = nn.MSELoss()(y_pred_train, y_train)
mse_loss_val = nn.MSELoss()(y_pred_val, y_val)
print("Final Training MSE:", mse_loss_train.item())
print("Final Validation MSE:", mse_loss_val.item())
y_pred_actual = y_pred_train * y_std + y_mean
y_actual = y_train * y_std + y_mean
ss_total = torch.sum((y_actual - torch.mean(y_actual)) ** 2)
ss_residual = torch.sum((y_actual - y_pred_actual) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(f"R² Score (unnormalized): {r2.item():.4f}")
print("Learned Weights:", model_multi.linear.weight.data.numpy())
print("Learned Bias:", model_multi.linear.bias.data.item())
plt.figure(figsize=(7, 5))
plt.plot(loss_multi, label="Multi-Feature Regression", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve: Multi-Feature Regression")
plt.legend()
save_current_figure("loss_curve_multi_feature.png")
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_actual.numpy().flatten(), y=(y_pred_actual).numpy().flatten(), alpha=0.5, label="Data Points")
sns.regplot(x=y_actual.numpy().flatten(), y=(y_pred_actual).numpy().flatten(),
            scatter=False, color='red', label="Regression Fit")
plt.xlabel("Actual Tip ($)")
plt.ylabel("Predicted Tip ($)")
plt.title(f"Multi-Feature Regression (Optimized), R² = {r2.item():.4f}")
plt.legend()
save_current_figure("actual_vs_predicted_multi_feature.png")

# ----- Polynomial Regression -----
df = load_dataset(FILE_PATH)
features = ["BillAmount", "NumItems", "Tip_Percentage"]
target_col = "Tip"
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(df, features, target_col, normalize=True)
degrees = [2, 4, 6]
results = {}
for degree in degrees:
    print(f"\n=== Polynomial Degree {degree} ===")
    X_poly = polynomial_features(X_tensor, degree)
    # Adjust learning rate and weight decay for degree 6 to improve stability
    lr_poly = 0.001 if degree < 6 else 0.0005
    wd_poly = 0.01 if degree < 6 else 0.05
    model_poly = LinearRegressionModel(input_dim=X_poly.shape[1])
    model_poly, loss_poly = train_regression_model(model_poly, X_poly, y_tensor,
                                                   learning_rate=lr_poly, epochs=1000,
                                                   print_every=200, weight_decay=wd_poly,
                                                   early_stopping_patience=200)
    with torch.no_grad():
        y_pred = model_poly(X_poly)
    mse_loss = nn.MSELoss()(y_pred, y_tensor)
    print(f"Final MSE (normalized): {mse_loss.item():.4f}")
    y_pred_actual = y_pred * y_std + y_mean
    y_actual = y_tensor * y_std + y_mean
    ss_total = torch.sum((y_actual - torch.mean(y_actual)) ** 2)
    ss_residual = torch.sum((y_actual - y_pred_actual) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print(f"R² Score (unnormalized): {r2.item():.4f}")
    results[degree] = {"MSE_normalized": mse_loss.item(), "R2_unnormalized": r2.item()}
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_actual.numpy().flatten(), y=y_pred_actual.numpy().flatten(), alpha=0.5, label="Data Points")
    sns.regplot(x=y_actual.numpy().flatten(), y=y_pred_actual.numpy().flatten(),
                scatter=False, color='red', label=f"Poly Degree {degree}")
    plt.xlabel("Actual Tip ($)")
    plt.ylabel("Predicted Tip ($)")
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.legend()
    save_current_figure(f"polynomial_regression_degree_{degree}.png")
print("\n=== Polynomial Regression Summary ===")
for deg, metrics in results.items():
    print(f"Degree {deg}: MSE (normalized) = {metrics['MSE_normalized']:.4f}, R² (unnormalized) = {metrics['R2_unnormalized']:.4f}")

# =============================================================================
# Part 4: Binary Classification (Logistic Regression vs. Neural Network)
# =============================================================================
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve, confusion_matrix
from itertools import combinations
from collections import Counter

# ----- Prepare Data for Binary Classification -----
df = load_dataset(FILE_PATH)
df["BinarySatisfaction"] = df["CustomerSatisfaction"].apply(lambda x: 1 if x == "Satisfied" else 0)
binary_features = ["BillAmount", "WaitTime", "NumItems", "Spending_per_Guest", "Is_Weekend", "Hour", "Tip_Percentage"]
binary_target = "BinarySatisfaction"
if df["Is_Weekend"].dtype == object:
    df["Is_Weekend"] = df["Is_Weekend"].apply(lambda x: 1 if str(x).lower() in ["yes", "true"] else 0)
if "DayOfWeek" in df.columns and df["DayOfWeek"].dtype == object:
    df["DayOfWeek"] = df["DayOfWeek"].astype("category").cat.codes
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
X_train, _, X_mean_bin, X_std_bin, _, _ = preprocess_data(train_df, binary_features, binary_target, normalize=True)
y_train = torch.tensor(train_df[binary_target].values, dtype=torch.float32).view(-1, 1)
X_test, _, _, _, _, _ = preprocess_data(test_df, binary_features, binary_target, normalize=True)
y_test = torch.tensor(test_df[binary_target].values, dtype=torch.float32).view(-1, 1)

def train_classification_model(model, X, y, learning_rate=0.001, epochs=500, print_every=50, weight_decay=0.01):
    """
    Train a binary classifier using BCEWithLogitsLoss.
    
    Returns:
      - Trained model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model

class LogisticRegressionClassifier(nn.Module):
    """
    Logistic Regression classifier with a single linear layer.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    """
    Neural network classifier with one hidden layer.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model_logistic = LogisticRegressionClassifier(input_dim=X_train.shape[1])
model_logistic = train_classification_model(model_logistic, X_train, y_train,
                                            learning_rate=0.001, epochs=500, print_every=50)
model_mlp = MLPClassifier(input_dim=X_train.shape[1])
model_mlp = train_classification_model(model_mlp, X_train, y_train,
                                       learning_rate=0.001, epochs=500, print_every=50)

def evaluate_classification_model(model, X, y, threshold=0.4):
    """
    Evaluate the binary classifier and return accuracy, precision, recall, F1-score, and probability estimates.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    accuracy = (preds.eq(y)).float().mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(y.numpy(), preds.numpy(), average="binary")
    return accuracy, precision, recall, f1, probs.numpy()

acc_log, prec_log, rec_log, f1_log, probs_log = evaluate_classification_model(model_logistic, X_test, y_test)
acc_mlp, prec_mlp, rec_mlp, f1_mlp, probs_mlp = evaluate_classification_model(model_mlp, X_test, y_test)
print("\nModel Performance Comparison (Binary Classification)")
print("-" * 40)
print(f"Logistic Regression - Accuracy: {acc_log:.4f}, Precision: {prec_log:.2f}, Recall: {rec_log:.2f}, F1-Score: {f1_log:.2f}")
print(f"Neural Network      - Accuracy: {acc_mlp:.4f}, Precision: {prec_mlp:.2f}, Recall: {rec_mlp:.2f}, F1-Score: {f1_mlp:.2f}")

fpr_log, tpr_log, _ = roc_curve(y_test.numpy(), probs_log)
roc_auc_log = auc(fpr_log, tpr_log)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test.numpy(), probs_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.figure(figsize=(7, 5))
plt.plot(fpr_log, tpr_log, color="blue", label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_mlp, tpr_mlp, color="red", label=f"Neural Network (AUC = {roc_auc_mlp:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Binary Classification)")
plt.legend()
save_current_figure("roc_curve_binary.png")

df_numeric_bin = df[binary_features + [binary_target]].select_dtypes(include=["number"])
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric_bin.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap (Binary Classification)")
save_current_figure("correlation_heatmap_binary.png")

# ----- Multi-Class Classification -----
df = load_dataset(FILE_PATH)
def categorize_tip(tip_value):
    if tip_value < 2.0:
        return 0  # Low Tip
    elif tip_value <= 5.0:
        return 1  # Medium Tip
    else:
        return 2  # High Tip
df["TipCategory"] = df["Tip"].apply(categorize_tip)
print("Unique classes in TipCategory:", df["TipCategory"].unique())
multi_features = ["BillAmount", "WaitTime", "NumItems", "Tip_Percentage", "Is_Weekend"]
multi_target = "TipCategory"
if df["Is_Weekend"].dtype == object:
    df["Is_Weekend"] = df["Is_Weekend"].apply(lambda x: 1 if str(x).lower() in ["yes", "true"] else 0)
train_df_multi = df.sample(frac=0.8, random_state=42)
test_df_multi = df.drop(train_df_multi.index)
from collections import Counter
class_counts = Counter(train_df_multi[multi_target])
print("Class Distribution Before Oversampling:", class_counts)
max_class_count = max(class_counts.values())
balanced_train_df = train_df_multi.copy()
for class_label, count in class_counts.items():
    if count < max_class_count:
        additional_samples = train_df_multi[train_df_multi[multi_target] == class_label].sample(
            n=(max_class_count - count), replace=True, random_state=42
        )
        balanced_train_df = pd.concat([balanced_train_df, additional_samples])
print("Class Distribution After Oversampling:", Counter(balanced_train_df[multi_target]))
X_train_multi, _, X_mean_multi, X_std_multi, _, _ = preprocess_data(balanced_train_df, multi_features, multi_target, normalize=True)
y_train_multi = torch.tensor(balanced_train_df[multi_target].values, dtype=torch.long)
X_test_multi, _, _, _, _, _ = preprocess_data(test_df_multi, multi_features, multi_target, normalize=True)
y_test_multi = torch.tensor(test_df_multi[multi_target].values, dtype=torch.long)
num_classes = len(df["TipCategory"].unique())
print(f"Number of classes: {num_classes}")

class LogisticRegressionMultiClassifier(nn.Module):
    """
    Logistic Regression model for multi-class classification (binary per class).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

def train_multi_logistic_model(model, X, y, lr=0.001, epochs=500, print_every=100):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model

def train_ova_classifiers(X, y, num_classes, lr=0.001, epochs=500):
    models = []
    for k in range(num_classes):
        print(f"\nTraining OvA Classifier for class {k} vs. others")
        y_k = (y == k).long()
        model_k = LogisticRegressionMultiClassifier(input_dim=X.shape[1])
        model_k = train_multi_logistic_model(model_k, X, y_k, lr=lr, epochs=epochs, print_every=100)
        models.append(model_k)
    return models

def predict_ova(models, X):
    model_logits = []
    with torch.no_grad():
        for mdl in models:
            logit = mdl(X)
            model_logits.append(logit)
    logits_stacked = torch.cat(model_logits, dim=1)
    preds = torch.argmax(logits_stacked, dim=1)
    return preds

print("\n===== One-vs-All (OvA) Multi-Class Classification =====")
ova_models = train_ova_classifiers(X_train_multi, y_train_multi, num_classes, lr=0.001, epochs=500)
ova_preds_test = predict_ova(ova_models, X_test_multi)
ova_accuracy = (ova_preds_test == y_test_multi).float().mean().item()
print(f"OvA Test Accuracy: {ova_accuracy:.4f}")

def train_ovo_classifiers(X, y, num_classes, lr=0.001, epochs=500):
    pairwise_models = {}
    from itertools import combinations
    for (i, j) in combinations(range(num_classes), 2):
        print(f"\nTraining OvO Classifier for classes {i} vs. {j}")
        mask_ij = (y == i) | (y == j)
        X_ij = X[mask_ij]
        y_ij = y[mask_ij]
        y_ij_binary = (y_ij == j).long()
        model_ij = LogisticRegressionMultiClassifier(input_dim=X.shape[1])
        model_ij = train_multi_logistic_model(model_ij, X_ij, y_ij_binary, lr=lr, epochs=epochs, print_every=100)
        pairwise_models[(i, j)] = model_ij
    return pairwise_models

def predict_ovo(pairwise_models, X, num_classes):
    votes = torch.zeros((X.shape[0], num_classes), dtype=torch.int)
    with torch.no_grad():
        for (i, j) in pairwise_models.keys():
            model_ij = pairwise_models[(i, j)]
            logits_ij = model_ij(X)
            preds_ij = (torch.sigmoid(logits_ij) >= 0.5).long()
            votes[:, i] += (preds_ij == 0).view(-1).int()
            votes[:, j] += (preds_ij == 1).view(-1).int()
    final_preds = torch.argmax(votes, dim=1)
    return final_preds

print("\n===== One-vs-One (OvO) Multi-Class Classification =====")
ovo_models = train_ovo_classifiers(X_train_multi, y_train_multi, num_classes, lr=0.001, epochs=500)
ovo_preds_test = predict_ovo(ovo_models, X_test_multi, num_classes)
ovo_accuracy = (ovo_preds_test == y_test_multi).float().mean().item()
print(f"OvO Test Accuracy: {ovo_accuracy:.4f}")

def plot_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_current_figure(filename)

plot_conf_matrix(y_test_multi, ova_preds_test, "OvA Confusion Matrix (Test)", "conf_matrix_ova.png")
plot_conf_matrix(y_test_multi, ovo_preds_test, "OvO Confusion Matrix (Test)", "conf_matrix_ovo.png")
