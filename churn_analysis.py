import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Loading and Initial Inspection
df = pd.read_csv('./data/spotify_churn_dataset.csv')

print("--- First 5 Rows of the Dataset ---")
print(df.head())
print("\n" + "="*50 + "\n")

print("--- DataFrame Info ---")
df.info()
print("\n" + "="*50 + "\n")

print("--- Descriptive Statistics ---")
print(df.describe())
print("\n" + "="*50 + "\n")

# 2. Data Cleaning and Feature Engineering
df.rename(columns={'is_churned': 'churn', 'listening_time': 'listening_hours'}, inplace=True)
df['premium_user'] = np.where(df['subscription_type'] == 'Free', 0, 1)
df_cleaned = df.drop('subscription_type', axis=1)
print("--- Data after Cleaning and Feature Engineering ---")
print(df_cleaned.head())
print("\n" + "="*50 + "\n")

# 3. Exploratory Data Analysis
sns.set_style('whitegrid')

# Calculate the overall churn rate
overall_churn_rate = np.mean(df_cleaned['churn'])
print(f"Overall Customer Churn Rate: {overall_churn_rate:.2%}\n")

# Analyze churn rates across different categories
print("--- Churn Rate by Premium Status ---")
churn_by_premium = df_cleaned.groupby('premium_user')['churn'].value_counts(normalize=True).unstack()
print(churn_by_premium)
print("\n")

print("--- Churn Rate by Country (Top 5) ---")
churn_by_country = df_cleaned.groupby('country')['churn'].value_counts(normalize=True).unstack()
print(churn_by_country.sort_values(by=1, ascending=False).head())
print("\n" + "="*50 + "\n")

# 4. Visualization
# Visualization 1: Distribution of the Target Variable 'churn'
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=df_cleaned, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')
plt.ylabel('Number of Customers')
plt.show()

# Visualization 2: Churn Rate by Premium User Status
plt.figure(figsize=(10, 7))
sns.countplot(x='premium_user', hue='churn', data=df_cleaned, palette='magma')
plt.title('Churn Comparison: Premium vs. Non-Premium Users')
plt.xlabel('Premium User Status')
plt.xticks([0, 1], ['Non-Premium', 'Premium'])
plt.ylabel('Number of Customers')
plt.legend(title='Churn', labels=['Retained', 'Churned'])
plt.show()

# Visualization 3: Listening Hours and Churn
plt.figure(figsize=(10, 7))
sns.boxplot(x='churn', y='listening_hours', data=df_cleaned, palette='coolwarm')
plt.title('Listening Hours Distribution by Churn Status')
plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')
plt.ylabel('Listening Hours')
plt.show()

# Visualization 4: Age and Churn
plt.figure(figsize=(10, 7))
sns.boxplot(x='churn', y='age', data=df_cleaned, palette='crest')
plt.title('Age Distribution by Churn Status')
plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')
plt.ylabel('Age')
plt.show()

# 5. Data Preprocessing for Modeling 
X = df_cleaned.drop(['churn', 'user_id'], axis=1)
y = df_cleaned['churn']

# Define features (X) and target (y)
# Drop user_id as it's an identifier and not a predictive feature
X = df_cleaned.drop(['churn', 'user_id'], axis=1)
y = df_cleaned['churn']

# One-hot encode categorical features to convert them into a numerical format
X = pd.get_dummies(X, columns=['gender', 'country', 'device_type'], drop_first=True)

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\n" + "="*50 + "\n")

# 7. Model Building and Evaluation
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)

# Print the final evaluation metrics
print("--- Logistic Regression Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
