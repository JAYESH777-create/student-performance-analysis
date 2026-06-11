# Step 1 - Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 2 - Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Step 3 - Look at the data
print("First 5 rows of data:")
print(df.head())

print("\nShape of dataset (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nBasic statistics:")
print(df.describe())

# Step 4 - Visualizations

# Chart 1: Average scores by gender
df.groupby('gender')[['math score','reading score','writing score']].mean().plot(kind='bar', color=['#4C72B0','#55A868','#C44E52'])
plt.title('Average Scores by Gender')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart1_gender.png')
plt.show()
plt.close()

# Chart 2: Effect of test preparation course
df.groupby('test preparation course')[['math score','reading score','writing score']].mean().plot(kind='bar', color=['#4C72B0','#55A868','#C44E52'])
plt.title('Impact of Test Preparation Course on Scores')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart2_testprep.png')
plt.show()
plt.close()

# Chart 3: Math score distribution
plt.figure(figsize=(8,5))
sns.histplot(df['math score'], bins=20, color='#4C72B0', kde=True)
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.tight_layout()
plt.savefig('chart3_distribution.png')
plt.show()
plt.close()

# Step 5 - Machine Learning Model
df_ml = df.copy()
le = LabelEncoder()
df_ml['gender'] = le.fit_transform(df_ml['gender'])
df_ml['test preparation course'] = le.fit_transform(df_ml['test preparation course'])
df_ml['lunch'] = le.fit_transform(df_ml['lunch'])

# Define input (X) and output (y)
X = df_ml[['gender', 'lunch', 'test preparation course', 'reading score', 'writing score']]
y = df_ml['math score']

# Split data — 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Results
print("\n--- ML Model Results ---")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print("\nWhat this means:")
print("R² Score tells how well the model predicts math scores.")
print("1.0 = perfect, 0.0 = terrible")