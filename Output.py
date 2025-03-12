# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# (Optional) For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Path to the cleaned dataset
FILE_PATH = "C:\\Users\\bramo\\Python Projects\\CAP4770-A2\\cleaned_restaurant_orders.csv"

# Load dataset
df = pd.read_csv(FILE_PATH)

# Ensure dataset contains required columns
expected_columns = ["BillAmount", "Spending_per_Guest", "WaitTime", "Tip"]
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    raise KeyError(f"⚠️ ERROR: Missing columns in dataset: {missing_columns}")
else:
    print("✅ Dataset loaded successfully. All expected columns are present.")

# Set Matplotlib style
sns.set_style("whitegrid")

##############################
#          Utilities
##############################

def load_dataset(file_path):
    """Load CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

# Ensure df is numeric before applying heatmap
df_numeric = df.select_dtypes(include=['number'])

# Verify df_numeric is not empty
if not df_numeric.empty:
    # Feature Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
else:
    print("⚠️ No numeric features found for correlation matrix.")

# Scatter Plots for Key Relationships (Tip vs. Key Features)
selected_features = ["BillAmount", "Spending_per_Guest", "WaitTime"]
if "Tip" in df.columns:
    sns.pairplot(df, x_vars=selected_features, y_vars=["Tip"], kind="scatter")
    plt.suptitle("Scatter Plots of Features vs. Tip", y=1.02)
    plt.show()
else:
    print("⚠️ 'Tip' column not found. Skipping scatter plot.")

def preprocess_data(df, feature_cols, target_col, normalize=True):
    """
    Extract features and target, convert to tensors, and optionally normalize.
    Returns:
      - X_tensor
      - y_tensor
      - X_mean, X_std (or None, None if normalize=False)
      - y_mean, y_std (or None, None if normalize=False)
    """
    if target_col not in df.columns:
        raise KeyError(f"⚠️ ERROR: Column '{target_col}' not found in dataset!")

    X = df[feature_cols].values
    y = df[[target_col]].values  # keep target as column vector
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    if normalize:
        X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0)
        # Guard against std=0
        X_std[X_std == 0] = 1.0
        
        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        if y_std == 0:
            y_std = 1.0
        
        X_tensor = (X_tensor - X_mean) / X_std
        y_tensor = (y_tensor - y_mean) / y_std
        
        return X_tensor, y_tensor, X_mean, X_std, y_mean, y_std
    else:
        return X_tensor, y_tensor, None, None, None, None

def plot_regression_line(
    X_orig, y_orig, 
    X_tensor, model, 
    X_mean, X_std, y_mean, y_std,
    title="Regression Plot", 
    xlabel="Feature", 
    ylabel="Target"
):
    """
    Plot the original data and the best-fit regression line.
    Converts normalized predictions back to original scale.
    """
    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).detach().flatten()
    
    # Convert predictions back to original scale if normalization was used
    y_pred_orig = predicted * y_std + y_mean
    
    # Sort for a smooth line
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
    plt.show()

def polynomial_features(X, degree):
    """
    Given a tensor X of shape (n_samples, n_features), return a new tensor 
    with each feature raised to powers 1 through 'degree' (no cross-terms). 
    Output shape: (n_samples, n_features * degree).
    """
    poly_terms = [X ** d for d in range(1, degree + 1)]
    return torch.cat(poly_terms, dim=1)


##############################
#       Models & Training
##############################

class LinearRegressionModel(nn.Module):
    """
    Simple Linear Regression Model that can handle 
    one or multiple input features.
    """
    def __init__(self, input_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def train_model(
    model, 
    X, 
    y, 
    learning_rate=0.001, 
    epochs=1000, 
    print_every=100,
    weight_decay=0.0, 
    early_stopping_patience=None,
    optimizer=None  
):
    """
    Train the model using MSE loss.
    - Supports Adam, SGD, or any optimizer.
    - Uses optional early stopping.
    """
    criterion = nn.MSELoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=0.5, verbose=False
    )
    
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
                print(f"[Early Stopping] Epoch={epoch}, Loss={loss.item():.4f}")
                break
        
        if (epoch + 1) % print_every == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    return model, loss_history  # ✅ Now returning both the trained model and loss values



# %% [markdown]
# # Part 1: Single-Feature Linear Regression
# 
# In this part, we will pick "BillAmount" as our single feature to predict "Tip". 
# We will demonstrate the training process, final parameters, final loss, 
# and plot the regression line.

# %%
# Part 1 - Single Feature Regression (Comparing Adam vs. SGD)

# 1) Load the dataset
df = load_dataset(FILE_PATH)

# 2) Define single feature and target
feature_cols = ["BillAmount"]
target_col = "Tip"

# Ensure 'Tip' column exists before proceeding
if target_col not in df.columns:
    raise KeyError(f"⚠️ ERROR: '{target_col}' column is missing from dataset!")

# 3) Preprocess data (normalize=True)
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(
    df, feature_cols, target_col, normalize=True
)

# 4) Initialize Single-Feature Linear Regression Models
model_adam = LinearRegressionModel(input_dim=1)
model_sgd = LinearRegressionModel(input_dim=1)

# 5) Define Optimizers
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)

# Train both models and store loss history
model_adam, loss_adam = train_model(
    model_adam,
    X_tensor,
    y_tensor,
    learning_rate=0.001,
    epochs=1000,
    print_every=100,
    weight_decay=0.0,
    early_stopping_patience=None,
    optimizer=optimizer_adam  # Now supported
)

model_sgd, loss_sgd = train_model(
    model_sgd,
    X_tensor,
    y_tensor,
    learning_rate=0.001,
    epochs=1000,
    print_every=100,
    weight_decay=0.0,
    early_stopping_patience=None,
    optimizer=optimizer_sgd  # Now supported
)


# 7) Evaluate final loss (MSE) for both models
with torch.no_grad():
    y_pred_adam = model_adam(X_tensor)
    mse_loss_adam = nn.MSELoss()(y_pred_adam, y_tensor)

    y_pred_sgd = model_sgd(X_tensor)
    mse_loss_sgd = nn.MSELoss()(y_pred_sgd, y_tensor)

print(f"✅ Final MSE (Adam, normalized): {mse_loss_adam.item():.4f}")
print(f"✅ Final MSE (SGD, normalized): {mse_loss_sgd.item():.4f}")

# 8) Convert predictions to original scale and compute R²
y_pred_actual_adam = y_pred_adam * y_std + y_mean
y_pred_actual_sgd = y_pred_sgd * y_std + y_mean
y_actual = y_tensor * y_std + y_mean

ss_total = torch.sum((y_actual - torch.mean(y_actual))**2)

ss_residual_adam = torch.sum((y_actual - y_pred_actual_adam)**2)
r2_adam = 1 - (ss_residual_adam / ss_total)

ss_residual_sgd = torch.sum((y_actual - y_pred_actual_sgd)**2)
r2_sgd = 1 - (ss_residual_sgd / ss_total)

print(f"✅ R² Score (Adam, unnormalized): {r2_adam.item():.4f}")
print(f"✅ R² Score (SGD, unnormalized): {r2_sgd.item():.4f}")

# 9) Plot Loss Curves to Compare Convergence Speed
plt.figure(figsize=(7, 5))
plt.plot(loss_adam, label="Adam", color="blue")
plt.plot(loss_sgd, label="SGD + Momentum", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve: Adam vs. SGD")
plt.legend()
plt.show()

# 10) Plot regression lines for both models
X_orig = df[feature_cols[0]].values
y_orig = df[target_col].values

plot_regression_line(
    X_orig,
    y_orig,
    X_tensor,
    model_adam,
    X_mean,
    X_std,
    y_mean,
    y_std,
    title="Part 1: Single-Feature Regression (Adam)",
    xlabel="Bill Amount ($)",
    ylabel="Tip ($)"
)

plot_regression_line(
    X_orig,
    y_orig,
    X_tensor,
    model_sgd,
    X_mean,
    X_std,
    y_mean,
    y_std,
    title="Part 1: Single-Feature Regression (SGD + Momentum)",
    xlabel="Bill Amount ($)",
    ylabel="Tip ($)"
)


# %% [markdown]
# # Part 2: Multiple-Feature Linear Regression
# 
# Here, we’ll use three features: BillAmount, NumItems and Spending_per_Guest

# %%
##############################
#   Part 2 - Multi-Feature Regression (Optimized)
##############################

# 1) Load the dataset
df = load_dataset(FILE_PATH)

# 2) Specify the best three features based on correlation
features = ["BillAmount", "NumItems", "Tip_Percentage"]  # ✅ Optimized features
target_col = "Tip"

# 3) Preprocess data (normalize=True)
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(
    df, features, target_col, normalize=True
)

# Print feature scaling values to verify normalization
print(f"X Mean: {X_mean.numpy()}, X Std: {X_std.numpy()}")
print(f"y Mean: {y_mean.numpy()}, y Std: {y_std.numpy()}")

# 4) Initialize the optimized multi-feature Linear Regression model
model_multi = LinearRegressionModel(input_dim=len(features))

# Print model type before training (debugging potential tuple error)
print("Before Training, Model Type:", type(model_multi))

# 5) Train-Test Split (80% Training, 20% Validation)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=0)

# 6) Train the model (Unpacking model and loss history correctly)
model_multi, loss_multi = train_model(  # ✅ FIXED unpacking issue
    model_multi,
    X_train,
    y_train,
    learning_rate=0.0003,  # ✅ Lower learning rate for stability
    epochs=2000,  # ✅ Increased training time for better results
    print_every=200,
    weight_decay=0.001  # L2 Regularization to prevent overfitting
)

# Print model type after training (debugging potential tuple error)
print("After Training, Model Type:", type(model_multi))  # ✅ Should now be a model, NOT a tuple!

# 7) Evaluate final MSE on normalized scale (Validation Set)
with torch.no_grad():
    y_pred_train = model_multi(X_train)
    y_pred_val = model_multi(X_val)

mse_loss_train = nn.MSELoss()(y_pred_train, y_train)
mse_loss_val = nn.MSELoss()(y_pred_val, y_val)

print(f"✅ Final Training MSE: {mse_loss_train.item():.4f}")
print(f"✅ Final Validation MSE: {mse_loss_val.item():.4f}")  # ✅ Helps detect overfitting

# 8) Convert predictions back to the original scale and compute R²
y_pred_actual = y_pred_train * y_std + y_mean
y_actual = y_train * y_std + y_mean

ss_total = torch.sum((y_actual - torch.mean(y_actual))**2)
ss_residual = torch.sum((y_actual - y_pred_actual) ** 2)  # ✅ Corrected squaring
r2 = 1 - (ss_residual / ss_total)

print(f"✅ R² Score (unnormalized): {r2.item():.4f}")  # ✅ Should be more reasonable now

# 9) Print out learned weights and bias
print("Learned Weights:", model_multi.linear.weight.data.numpy())
print("Learned Bias:", model_multi.linear.bias.data.item())

# 10) Plot Training Loss for Multi-Feature Regression
plt.figure(figsize=(7, 5))
plt.plot(loss_multi, label="Multi-Feature Regression", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve: Multi-Feature Regression")
plt.legend()
plt.show()

# 11) Plot Actual vs. Predicted Tip (unnormalized)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_actual.numpy().flatten(), y=y_pred_actual.numpy().flatten(), alpha=0.5, label="Data Points")
sns.regplot(
    x=y_actual.numpy().flatten(),
    y=y_pred_actual.numpy().flatten(),
    scatter=False, 
    color='red', 
    label="Regression Fit"
)
plt.xlabel("Actual Tip ($)")
plt.ylabel("Predicted Tip ($)")
plt.title(f"Part 2: Multi-Feature Regression (Optimized), R² = {r2.item():.4f}")
plt.legend()
plt.show()


# %% [markdown]
# # Part 3: Polynomial Regression
# 
# We compare different polynomial degrees (2, 4, 6) using the same 3 features.

# %%
##############################
#   Part 3 - Polynomial Regression
##############################

# 1) Load the dataset
df = load_dataset(FILE_PATH)

# 2) Define the best three features and the target
features = ["BillAmount", "NumItems", "Tip_Percentage"]  # ✅ Updated feature selection
target_col = "Tip"

# 3) Preprocess data (normalize=True)
X_tensor, y_tensor, X_mean, X_std, y_mean, y_std = preprocess_data(
    df, features, target_col, normalize=True
)

# We will test 3 polynomial degrees
degrees = [2, 4, 6]

# Dictionary to store final MSE & R² results for each degree
results = {}

for degree in degrees:
    print(f"\n=== Polynomial Degree {degree} ===")
    
    # 4) Generate polynomial features (no cross-terms)
    X_poly = polynomial_features(X_tensor, degree)
    
    # 5) Initialize and train a linear model on X_poly
    model_poly = LinearRegressionModel(input_dim=X_poly.shape[1])
    model_poly, loss_poly = train_model(  # ✅ FIXED unpacking issue
        model_poly,
        X_poly,
        y_tensor,
        learning_rate=0.001,
        epochs=1000,
        print_every=200,
        weight_decay=0.01,   # ✅ Optional small L2 regularization to mitigate overfitting
        early_stopping_patience=200  # ✅ Optional patience for early stopping
    )
    
    # 6) Evaluate final MSE (on normalized scale)
    with torch.no_grad():
        y_pred = model_poly(X_poly)

    mse_loss = nn.MSELoss()(y_pred, y_tensor)
    print(f"✅ Final MSE (normalized): {mse_loss.item():.4f}")
    
    # Convert predictions back to original scale to compute R²
    y_pred_actual = y_pred * y_std + y_mean
    y_actual = y_tensor * y_std + y_mean
    
    ss_total = torch.sum((y_actual - torch.mean(y_actual))**2)
    ss_residual = torch.sum((y_actual - y_pred_actual) ** 2)  # ✅ Corrected squaring issue
    r2 = 1 - (ss_residual / ss_total)
    print(f"✅ R² Score (unnormalized): {r2.item():.4f}")
    
    # Store in our results dict
    results[degree] = {
        "MSE_normalized": mse_loss.item(),
        "R2_unnormalized": r2.item()
    }
    
    # 7) Plot Actual vs. Predicted
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=y_actual.numpy().flatten(), 
        y=y_pred_actual.numpy().flatten(),
        alpha=0.5, label="Data Points"
    )
    sns.regplot(
        x=y_actual.numpy().flatten(),
        y=y_pred_actual.numpy().flatten(),
        scatter=False, color='red',
        label=f"Poly Deg={degree}"
    )
    plt.xlabel("Actual Tip ($)")
    plt.ylabel("Predicted Tip ($)")
    plt.title(f"Part 3: Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.show()

# 8) Print overall summary of results
print("\n=== Polynomial Regression Summary ===")
for deg, metrics in results.items():
    print(f"Degree={deg}: MSE (norm)={metrics['MSE_normalized']:.4f}, R²(unnorm)={metrics['R2_unnormalized']:.4f}")


# %% [markdown]
# # Part 4: Binary Classification
# 
# We create a binary label (Satisfied vs. Unsatisfied) and train a logistic regression.

# %%
# Part 4 - Binary Classification (Comparing Logistic Regression vs. Neural Network)

##############################
# 1) Load & Prepare the Data
##############################
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve


df = load_dataset(FILE_PATH)

# Create a binary label: 1 for "Satisfied", 0 for anything else
df["BinarySatisfaction"] = df["CustomerSatisfaction"].apply(lambda x: 1 if x == "Satisfied" else 0)

# ✅ Features Used (Same for Both Models)
features = ["BillAmount", "WaitTime", "NumItems", "Spending_per_Guest", "Is_Weekend", "Hour", "Tip_Percentage"]
target_col = "BinarySatisfaction"

# Convert categorical variables if needed
if df["Is_Weekend"].dtype == object:
    df["Is_Weekend"] = df["Is_Weekend"].apply(lambda x: 1 if str(x).lower() in ["yes", "true"] else 0)

# Convert "DayOfWeek" to numerical representation (Optional, if categorical)
if "DayOfWeek" in df.columns and df["DayOfWeek"].dtype == object:
    df["DayOfWeek"] = df["DayOfWeek"].astype("category").cat.codes

# Train/test split (80/20)
train_df = df.sample(frac=0.8, random_state=42)
test_df  = df.drop(train_df.index)

##############################
# 2) Preprocess Features
##############################
# We'll normalize the features, but NOT the target
X_train, _, X_mean, X_std, _, _ = preprocess_data(
    train_df, 
    features, 
    target_col, 
    normalize=True
)
y_train = torch.tensor(train_df[target_col].values, dtype=torch.float32).view(-1, 1)

X_test, _, _, _, _, _ = preprocess_data(
    test_df, 
    features, 
    target_col, 
    normalize=True
)
y_test = torch.tensor(test_df[target_col].values, dtype=torch.float32).view(-1, 1)

##############################
# 3) Model 1: Logistic Regression (Allowed by Guidelines)
##############################
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # ✅ Only One Linear Layer (Allowed)

    def forward(self, x):
        return self.linear(x)  # ✅ No Activation, BCEWithLogitsLoss expects logits

def train_model(
    model, 
    X, 
    y, 
    learning_rate=0.001, 
    epochs=500, 
    print_every=50,
    weight_decay=0.01  # ✅ L2 Regularization (Allowed)
):
    """
    Train a binary classifier using BCEWithLogitsLoss.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X)                      
        loss = criterion(logits, y)  # BCE loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

##############################
# 4) Model 2: Neural Network (For Comparison)
##############################
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)  # Hidden layer with 8 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)  # Output layer
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

##############################
# 5) Train Both Models & Compare
##############################
# Train Logistic Regression
model_logistic = LogisticRegressionModel(input_dim=X_train.shape[1])
model_logistic = train_model(
    model_logistic, X_train, y_train,
    learning_rate=0.001, epochs=500, print_every=50
)

# Train Neural Network
model_mlp = MLPClassifier(input_dim=X_train.shape[1])
model_mlp = train_model(
    model_mlp, X_train, y_train,
    learning_rate=0.001, epochs=500, print_every=50
)

##############################
# 6) Evaluate Both Models
##############################
def evaluate_model(model, X, y, threshold=0.4):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)  

    preds = (probs >= threshold).float()  # ✅ Adjusted threshold for better recall
    accuracy = (preds.eq(y)).float().mean().item()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.numpy(), preds.numpy(), average="binary"
    )

    return accuracy, precision, recall, f1, probs.numpy()

# Evaluate Logistic Regression
acc_log, prec_log, rec_log, f1_log, probs_log = evaluate_model(model_logistic, X_test, y_test)

# Evaluate Neural Network
acc_mlp, prec_mlp, rec_mlp, f1_mlp, probs_mlp = evaluate_model(model_mlp, X_test, y_test)

# Print comparison results
print("\nModel Performance Comparison")
print("-" * 40)
print(f"Logistic Regression - Accuracy: {acc_log:.4f}, Precision: {prec_log:.2f}, Recall: {rec_log:.2f}, F1-Score: {f1_log:.2f}")
print(f"Neural Network      - Accuracy: {acc_mlp:.4f}, Precision: {prec_mlp:.2f}, Recall: {rec_mlp:.2f}, F1-Score: {f1_mlp:.2f}")

##############################
# 7) Plot ROC Curves, Heatmap & Correlation Matrix
##############################
# Plot ROC Curves
fpr_log, tpr_log, _ = roc_curve(y_test.numpy(), probs_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_mlp, tpr_mlp, _ = roc_curve(y_test.numpy(), probs_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.figure(figsize=(7, 5))
plt.plot(fpr_log, tpr_log, color="blue", label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_mlp, tpr_mlp, color="red", label=f"Neural Network (AUC = {roc_auc_mlp:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Compute correlation matrix (including target variable)
df_numeric = df[features + ["BinarySatisfaction"]].select_dtypes(include=["number"])
corr_matrix = df_numeric.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap (Including Binary Target)")
plt.show()



# %% [markdown]
# # Part 5: Multiple-Class Classification
# 
# We will create a 3-class label (e.g., 'Low Tip', 'Medium Tip', 'High Tip') or 
# use 'CustomerSatisfaction' if it has 3 categories, and implement One-vs-All logistic regression.

# %%
##############################
#   Part 5 - Multi-Class Classification (OvA & OvO)
##############################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from itertools import combinations
from collections import Counter

##############################
# 1) Load & Prepare the Data
##############################
df = load_dataset(FILE_PATH)

# Categorize tips into 3 classes: Low, Medium, High
def categorize_tip(tip_value):
    if tip_value < 2.0:
        return 0  # "Low Tip"
    elif tip_value <= 5.0:
        return 1  # "Medium Tip"
    else:
        return 2  # "High Tip"

df["TipCategory"] = df["Tip"].apply(categorize_tip)
print("Unique classes in TipCategory:", df["TipCategory"].unique())

# **Updated Feature Selection** (Replaced Spending_per_Guest with Tip_Percentage)
feature_cols = ["BillAmount", "WaitTime", "NumItems", "Tip_Percentage", "Is_Weekend"]
target_col = "TipCategory"

# Ensure Is_Weekend is numeric
if df["Is_Weekend"].dtype == object:
    df["Is_Weekend"] = df["Is_Weekend"].apply(lambda x: 1 if str(x).lower() in ["yes","true"] else 0)

# Train/Test Split (80/20)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Balance the classes (Oversampling if needed)
class_counts = Counter(train_df[target_col])
print("Class Distribution Before Oversampling:", class_counts)

max_class_count = max(class_counts.values())
balanced_train_df = train_df.copy()

for class_label, count in class_counts.items():
    if count < max_class_count:
        additional_samples = train_df[train_df[target_col] == class_label].sample(
            n=(max_class_count - count), replace=True, random_state=42
        )
        balanced_train_df = pd.concat([balanced_train_df, additional_samples])

print("Class Distribution After Oversampling:", Counter(balanced_train_df[target_col]))

# Preprocess features (normalize)
X_train, _, X_mean, X_std, _, _ = preprocess_data(balanced_train_df, feature_cols, target_col, normalize=True)
y_train = torch.tensor(balanced_train_df[target_col].values, dtype=torch.long)  # Multi-class labels (long type)
X_test, _, _, _, _, _ = preprocess_data(test_df, feature_cols, target_col, normalize=True)
y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

num_classes = len(df["TipCategory"].unique())
print(f"Number of classes: {num_classes}")

##############################
# 2) One-vs-All (OvA) Training
##############################
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # Outputs raw logits

def train_logistic_model(model, X, y, lr=0.001, epochs=500, print_every=100):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X)
        loss = criterion(logits, y.view(-1, 1).float())  # BCE requires float [N,1]
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def train_ova_classifiers(X, y, num_classes, lr=0.001, epochs=500):
    models = []
    for k in range(num_classes):
        print(f"\n--- Training OvA Classifier for class={k} vs. others ---")
        y_k = (y == k).long()
        model_k = LogisticRegressionModel(input_dim=X.shape[1])
        model_k = train_logistic_model(model_k, X, y_k, lr=lr, epochs=epochs, print_every=100)
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

print("\n============= One-vs-All (OvA) =============")
ova_models = train_ova_classifiers(X_train, y_train, num_classes, lr=0.001, epochs=500)
ova_preds_test = predict_ova(ova_models, X_test)
ova_accuracy = (ova_preds_test == y_test).float().mean().item()
print(f"OvA Test Accuracy: {ova_accuracy:.4f}")

##############################
# 3) One-vs-One (OvO) Training
##############################
def train_ovo_classifiers(X, y, num_classes, lr=0.001, epochs=500):
    pairwise_models = {}
    for (i, j) in combinations(range(num_classes), 2):
        print(f"\n--- Training OvO Classifier for classes={i} vs. {j} ---")
        mask_ij = (y == i) | (y == j)
        X_ij = X[mask_ij]
        y_ij = y[mask_ij]
        y_ij_binary = (y_ij == j).long()
        
        model_ij = LogisticRegressionModel(input_dim=X.shape[1])
        model_ij = train_logistic_model(model_ij, X_ij, y_ij_binary, lr=lr, epochs=epochs, print_every=100)
        pairwise_models[(i,j)] = model_ij
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

print("\n============= One-vs-One (OvO) =============")
ovo_models = train_ovo_classifiers(X_train, y_train, num_classes, lr=0.001, epochs=500)
ovo_preds_test = predict_ovo(ovo_models, X_test, num_classes)
ovo_accuracy = (ovo_preds_test == y_test).float().mean().item()
print(f"OvO Test Accuracy: {ovo_accuracy:.4f}")

##############################
# 4) Confusion Matrices
##############################
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_test, ova_preds_test, "OvA Confusion Matrix (Test)")
plot_conf_matrix(y_test, ovo_preds_test, "OvO Confusion Matrix (Test)")



