import pandas as pd

# Load the dataset
file_path = "C:\\Users\\bramo\\Python Projects\\CAP4770-A1\\restaurant_orders.csv"
df = pd.read_csv(file_path)

# Convert OrderDateTime to datetime format
df["OrderDateTime"] = pd.to_datetime(df["OrderDateTime"])

# Step 1: Identifying Issues in the Data
## Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

## Checking for duplicates
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

# Step 2: Handling Outliers (Using IQR Method)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_clean = ["BillAmount", "Tip", "WaitTime"]
for col in columns_to_clean:
    df = remove_outliers_iqr(df, col)

# Step 3: Handling Missing Values (Impute WaitTime using Median)
df["WaitTime"] = df["WaitTime"].fillna(df["WaitTime"].median())

# Step 4: Checking Logical Inconsistencies
## Extracting number of items ordered
def count_items(ordered_items):
    if isinstance(ordered_items, str):
        items_list = ordered_items.strip("[]").split(",")
        return len(items_list)
    return 0

df["NumItems"] = df["ItemsOrdered"].apply(count_items)
## Identifying rows where NumberOfGuests > NumItems
inconsistent_rows = df[df["NumberOfGuests"] > df["NumItems"]]
print("Logical Inconsistencies:\n", inconsistent_rows)

# Step 5: Feature Engineering
## Tip Percentage
df["Tip_Percentage"] = (df["Tip"] / df["BillAmount"]) * 100
## Spending per Guest
df["Spending_per_Guest"] = df["BillAmount"] / df["NumberOfGuests"]
## Peak Hour Category
def categorize_hour(hour):
    if 11 <= hour <= 14:
        return "Lunch"
    elif 18 <= hour <= 21:
        return "Dinner"
    else:
        return "Off-Peak"

df["Hour"] = df["OrderDateTime"].dt.hour
df["Peak_Hour_Category"] = df["Hour"].apply(categorize_hour)
## Weekday vs Weekend Indicator
df["DayOfWeek"] = df["OrderDateTime"].dt.day_name()
df["Is_Weekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in ["Saturday", "Sunday"] else 0)

# Save the cleaned dataset
output_path = "C:\\Users\\bramo\\Python Projects\\CAP4770-A1\\cleaned_restaurant_orders.csv"
df.to_csv(output_path, index=False)
print(f"Preprocessing complete. Cleaned dataset saved as '{output_path}'")