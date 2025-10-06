# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)

# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# =============================================================================

print("\nSTEP 1: Loading and Exploring Data")
print("-" * 40)

# Load the dataset
try:
    df = pd.read_csv('house_data.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: house_data.csv not found!")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'Id': range(1, n_samples + 1),
        'Area': np.random.normal(2000, 500, n_samples).astype(int),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.randint(1, 4, n_samples),
        'Floors': np.random.randint(1, 4, n_samples),
        'YearBuilt': np.random.randint(1950, 2023, n_samples),
        'Location': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
        'Condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
        'Garage': np.random.choice(['No', 'Yes'], n_samples)
    })
    
    # Generate realistic prices based on features
    price_base = (df['Area'] * 100 + 
                  df['Bedrooms'] * 10000 + 
                  df['Bathrooms'] * 15000 + 
                  (2023 - df['YearBuilt']) * -500)
    
    location_multiplier = df['Location'].map({'Rural': 0.8, 'Suburb': 1.0, 'Downtown': 1.3})
    condition_multiplier = df['Condition'].map({'Poor': 0.7, 'Fair': 0.85, 'Good': 1.0, 'Excellent': 1.2})
    garage_multiplier = df['Garage'].map({'No': 0.95, 'Yes': 1.05})
    
    df['Price'] = (price_base * location_multiplier * condition_multiplier * garage_multiplier + 
                   np.random.normal(0, 20000, n_samples)).astype(int)
    
    # Add some missing values for realism
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices[:20], 'Bathrooms'] = np.nan
    df.loc[missing_indices[20:35], 'YearBuilt'] = np.nan
    df.loc[missing_indices[35:], 'Garage'] = np.nan
    
    print("✅ Sample dataset created for demonstration!")

# Display basic information
print(f"\n📋 First 5 rows:")
print(df.head())

print(f"\n📊 Dataset Info:")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

print(f"\n📈 Statistical Summary:")
print(df.describe())

# =============================================================================
# PART 2: MISSING VALUES ANALYSIS AND VISUALIZATION
# =============================================================================

print("\nSTEP 2: Missing Values Analysis")
print("-" * 40)

# Check missing values
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_percent
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]

if len(missing_summary) > 0:
    print("Missing Values Found:")
    print(missing_summary)
else:
    print("✅ No missing values found!")

# Visualization 1: Missing Values Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap (Yellow = Missing)')
plt.xlabel('Columns')
plt.tight_layout()
plt.show()

# Handle missing values
print("\n🔧 Handling Missing Values...")
df_original = df.copy()

# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: Filled with median ({median_val})")

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col}: Filled with mode ({mode_val})")

# Visualization 2: Before/After Missing Values Treatment
affected_cols = [col for col in df.columns if df_original[col].isnull().sum() > 0]

if affected_cols:
    fig, axes = plt.subplots(len(affected_cols), 2, figsize=(15, 4*len(affected_cols)))
    if len(affected_cols) == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(affected_cols):
        # Before
        if df[col].dtype == 'object':
            df_original[col].value_counts().plot(kind='bar', ax=axes[i,0], color='red', alpha=0.7)
        else:
            axes[i,0].hist(df_original[col].dropna(), bins=20, color='red', alpha=0.7)
        axes[i,0].set_title(f'{col} - Before (Missing: {df_original[col].isnull().sum()})')
        
        # After
        if df[col].dtype == 'object':
            df[col].value_counts().plot(kind='bar', ax=axes[i,1], color='green', alpha=0.7)
        else:
            axes[i,1].hist(df[col], bins=20, color='green', alpha=0.7)
        axes[i,1].set_title(f'{col} - After (Missing: {df[col].isnull().sum()})')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 3: FEATURE ENGINEERING AND ENCODING
# =============================================================================

print("\nSTEP 3: Feature Engineering")
print("-" * 40)

# Create House Age feature
current_year = 2023
if 'YearBuilt' in df.columns:
    df['House_Age'] = current_year - df['YearBuilt']
    print(f"✅ Created House_Age feature")

# Encode categorical variables
print("\n🔤 Encoding Categorical Features...")
label_encoders = {}
categorical_features = ['Location', 'Condition', 'Garage']

for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le
        print(f"  {feature}: {len(le.classes_)} categories encoded")

# =============================================================================
# PART 4: OUTLIER DETECTION AND TREATMENT
# =============================================================================

print("\nSTEP 4: Outlier Detection and Treatment")
print("-" * 40)

outlier_features = ['Area', 'Bedrooms', 'Bathrooms', 'Price']
available_features = [f for f in outlier_features if f in df.columns]

# Visualization 3: Outlier Detection with Boxplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

outlier_stats = {}
for i, feature in enumerate(available_features[:4]):
    # Calculate outlier bounds
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    outlier_stats[feature] = len(outliers)
    
    # Create boxplot
    axes[i].boxplot(df[feature])
    axes[i].set_title(f'{feature} - Outliers: {len(outliers)}')
    axes[i].set_ylabel(feature)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Outlier Detection - Before Treatment', y=1.02)
plt.show()

# Handle outliers using IQR method (capping)
print("\n🔧 Treating Outliers...")
df_before_outlier = df.copy()

for feature in available_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = len(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)])
    
    # Cap outliers
    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
    
    outliers_after = len(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)])
    print(f"  {feature}: {outliers_before} outliers treated")

# Visualization 4: Before/After Outlier Treatment
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, feature in enumerate(available_features):
    # Before
    axes[0, i].boxplot(df_before_outlier[feature])
    axes[0, i].set_title(f'{feature} - Before')
    axes[0, i].grid(True, alpha=0.3)
    
    # After
    axes[1, i].boxplot(df[feature])
    axes[1, i].set_title(f'{feature} - After')
    axes[1, i].grid(True, alpha=0.3)
    
    # Add improvement stats
    before_std = df_before_outlier[feature].std()
    after_std = df[feature].std()
    improvement = ((before_std - after_std) / before_std) * 100
    axes[1, i].text(0.5, 0.95, f'Std reduced: {improvement:.1f}%', 
                   transform=axes[1, i].transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

# =============================================================================
# PART 5: CORRELATION ANALYSIS
# =============================================================================

print("\nSTEP 5: Correlation Analysis")
print("-" * 40)

# Select numeric columns for correlation
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Id' in numeric_columns:
    numeric_columns.remove('Id')

# Visualization 5: Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, fmt='.2f', mask=mask)
plt.title('Correlation Heatmap - Numeric Features')
plt.tight_layout()
plt.show()

# Print strongest correlations with Price
if 'Price' in correlation_matrix.columns:
    price_corr = correlation_matrix['Price'].abs().sort_values(ascending=False)
    print("\n🏆 Top Features Correlated with Price:")
    for feature, corr in price_corr.items():
        if feature != 'Price':
            print(f"  {feature}: {corr:.3f}")

# =============================================================================
# PART 6: EXPLORATORY DATA ANALYSIS VISUALIZATIONS
# =============================================================================

print("\nSTEP 6: Exploratory Data Analysis")
print("-" * 40)

# Visualization 6: Feature vs Price Scatterplots
scatter_features = ['Area', 'Bedrooms', 'Bathrooms', 'House_Age']
available_scatter = [f for f in scatter_features if f in df.columns]

if len(available_scatter) > 0 and 'Price' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_scatter[:4]):
        axes[i].scatter(df[feature], df['Price'], alpha=0.6, s=20)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Price')
        axes[i].set_title(f'{feature} vs Price')
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation
        if feature in df.columns:
            corr = df[feature].corr(df['Price'])
            axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Remove empty subplots
    for j in range(len(available_scatter), 4):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Visualization 7: Price Distribution
if 'Price' in df.columns:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['Price'].mean(), color='red', linestyle='--', 
               label=f'Mean: ${df["Price"].mean():,.0f}')
    plt.axvline(df['Price'].median(), color='green', linestyle='--', 
               label=f'Median: ${df["Price"].median():,.0f}')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='Price', ax=plt.gca())
    plt.title('Price Distribution - Boxplot')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 7: MODEL PREPARATION
# =============================================================================

print("\nSTEP 7: Model Preparation")
print("-" * 40)

# Prepare features and target
feature_columns = []
for col in df.columns:
    if (df[col].dtype in ['int64', 'float64'] and 
        col not in ['Price', 'Id'] and 
        not col.startswith('Year')):  # Exclude YearBuilt to avoid multicollinearity with House_Age
        feature_columns.append(col)

print(f"Selected Features ({len(feature_columns)}):")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")

# Prepare X and y
X = df[feature_columns].copy()
y = df['Price'].copy()

print(f"\nDataset dimensions: X={X.shape}, y={y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Preprocessing
print("\n🔧 Preprocessing...")

# Imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("✅ Preprocessing completed!")

# =============================================================================
# PART 8: MODEL TRAINING AND EVALUATION
# =============================================================================

print("\nSTEP 8: Model Training")
print("-" * 40)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

# Train and evaluate models
results = {}
print("Training models...")

for name, model in models.items():
    print(f"  Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'predictions': y_pred_test
    }
    
    print(f"    R²: {test_r2:.4f} | MAE: ${test_mae:,.0f}")

# =============================================================================
# PART 9: MODEL PERFORMANCE VISUALIZATION
# =============================================================================

print("\nSTEP 9: Model Performance Analysis")
print("-" * 40)

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train_R2': [results[name]['train_r2'] for name in results.keys()],
    'Test_R2': [results[name]['test_r2'] for name in results.keys()],
    'Test_MAE': [results[name]['test_mae'] for name in results.keys()],
    'Test_RMSE': [results[name]['test_rmse'] for name in results.keys()]
})

print("Model Performance Summary:")
print(results_df.round(4))

# Find best model
best_model_name = results_df.loc[results_df['Test_R2'].idxmax(), 'Model']
best_r2 = results_df['Test_R2'].max()

print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

# Visualization 8: Model Performance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# R² Scores
model_names = list(results.keys())
train_r2 = [results[name]['train_r2'] for name in model_names]
test_r2 = [results[name]['test_r2'] for name in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

ax1.bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8, color='lightblue')
ax1.bar(x_pos + width/2, test_r2, width, label='Test R²', alpha=0.8, color='darkblue')
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance - R² Scores')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels
for i, (train, test) in enumerate(zip(train_r2, test_r2)):
    ax1.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)

# Error Metrics
test_mae = [results[name]['test_mae'] for name in model_names]
test_rmse = [results[name]['test_rmse'] for name in model_names]

ax2.bar(x_pos - width/2, test_mae, width, label='MAE', alpha=0.8, color='lightcoral')
ax2.bar(x_pos + width/2, test_rmse, width, label='RMSE', alpha=0.8, color='darkred')
ax2.set_xlabel('Models')
ax2.set_ylabel('Error ($)')
ax2.set_title('Model Performance - Error Metrics')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualization 9: Residual Analysis
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, (name, result) in enumerate(results.items()):
    residuals = y_test - result['predictions']
    
    axes[i].scatter(result['predictions'], residuals, alpha=0.6, s=30)
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[i].set_xlabel('Predicted Price ($)')
    axes[i].set_ylabel('Residuals ($)')
    axes[i].set_title(f'{name} - Residual Plot')
    axes[i].grid(True, alpha=0.3)
    
    # Add stats
    r2 = result['test_r2']
    mae = result['test_mae']
    axes[i].text(0.05, 0.95, f'R²: {r2:.3f}\nMAE: ${mae:,.0f}', 
                transform=axes[i].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.suptitle('Residual Analysis for All Models', y=1.02)
plt.show()

# Visualization 10: Actual vs Predicted for Best Model
best_predictions = results[best_model_name]['predictions']

plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted - {best_model_name}')
plt.grid(True, alpha=0.3)

# Add R² on plot
plt.text(0.05, 0.95, f'R² = {best_r2:.4f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.show()

# =============================================================================
# PART 10: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\nSTEP 10: Feature Importance Analysis")
print("-" * 40)

# Get feature importance for tree-based models
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Random Forest Feature Importance:")
    print(feature_importance)
    
    # Visualization 11: Feature Importance Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['Importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(feature_importance['Importance']):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

# Linear model coefficients analysis
if 'Linear Regression' in results:
    linear_model = results['Linear Regression']['model']
    ridge_model = results['Ridge']['model']
    lasso_model = results['Lasso']['model']
    
    coeff_df = pd.DataFrame({
        'Feature': feature_columns,
        'Linear': linear_model.coef_,
        'Ridge': ridge_model.coef_,
        'Lasso': lasso_model.coef_
    })
    
    print("\nLinear Model Coefficients:")
    print(coeff_df.round(4))
    
    # Visualization 12: Coefficient Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    y_pos = np.arange(len(feature_columns))
    
    ax1.barh(y_pos, coeff_df['Linear'], alpha=0.7, color='blue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_columns)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title('Linear Regression Coefficients')
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.grid(True, alpha=0.3)
    
    ax2.barh(y_pos, coeff_df['Ridge'], alpha=0.7, color='green')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_columns)
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title('Ridge Regression Coefficients')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.grid(True, alpha=0.3)
    
    ax3.barh(y_pos, coeff_df['Lasso'], alpha=0.7, color='orange')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feature_columns)
    ax3.set_xlabel('Coefficient Value')
    ax3.set_title('Lasso Regression Coefficients')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Count non-zero Lasso coefficients
    non_zero_lasso = np.sum(np.abs(coeff_df['Lasso']) > 1e-10)
    print(f"\nLasso Feature Selection: {non_zero_lasso}/{len(feature_columns)} features selected")

# =============================================================================
# PART 11: MODEL PERSISTENCE
# =============================================================================

print("\nSTEP 11: Model Persistence")
print("-" * 40)

# Create directory for saved models
models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

print("Saving models and preprocessing components...")

# Save all models
for name, result in results.items():
    filename = name.replace(' ', '_').lower() + '_model.pkl'
    filepath = os.path.join(models_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(result['model'], f)
    print(f"✅ Saved {name}")

# Save preprocessing components
with open(os.path.join(models_dir, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)

with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(models_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feature_columns, f)

with open(os.path.join(models_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ All preprocessing components saved!")

# Save model performance results
results_summary = {
    'performance_df': results_df,
    'best_model_name': best_model_name,
    'best_r2_score': best_r2,
    'feature_importance': feature_importance if 'Random Forest' in results else None,
    'coefficients': coeff_df if 'Linear Regression' in results else None
}

with open(os.path.join(models_dir, 'results_summary.pkl'), 'wb') as f:
    pickle.dump(results_summary, f)

print("✅ Results summary saved!")

# =============================================================================
# PART 12: PREDICTION FUNCTION
# =============================================================================

def predict_house_price(area, bedrooms, bathrooms, floors, year_built, 
                       location, condition, garage, model_name='Random Forest'):
    """
    Predict house price using trained model
    
    Parameters:
    - area: House area in sq ft
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - floors: Number of floors
    - year_built: Year the house was built
    - location: Location ('Downtown', 'Suburb', 'Rural')
    - condition: Condition ('Poor', 'Fair', 'Good', 'Excellent')
    - garage: Garage ('Yes', 'No')
    - model_name: Name of model to use
    
    Returns:
    - Predicted price
    """
    try:
        # Load model and preprocessing components
        model_filename = model_name.replace(' ', '_').lower() + '_model.pkl'
        model_path = os.path.join(models_dir, model_filename)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'imputer.pkl'), 'rb') as f:
            loaded_imputer = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            loaded_features = pickle.load(f)
        
        with open(os.path.join(models_dir, 'label_encoders.pkl'), 'rb') as f:
            loaded_encoders = pickle.load(f)
        
        # Create feature dictionary
        house_age = 2023 - year_built
        
        # Encode categorical features
        location_encoded = loaded_encoders['Location'].transform([location])[0]
        condition_encoded = loaded_encoders['Condition'].transform([condition])[0]
        garage_encoded = loaded_encoders['Garage'].transform([garage])[0]
        
        # Create feature vector in the same order as training
        feature_dict = {
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'House_Age': house_age,
            'Location_encoded': location_encoded,
            'Condition_encoded': condition_encoded,
            'Garage_encoded': garage_encoded
        }
        
        # Create feature vector
        feature_vector = []
        for feature in loaded_features:
            if feature in feature_dict:
                feature_vector.append(feature_dict[feature])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        X_new = np.array(feature_vector).reshape(1, -1)
        
        # Apply preprocessing
        X_new_imputed = loaded_imputer.transform(X_new)
        X_new_scaled = loaded_scaler.transform(X_new_imputed)
        
        # Make prediction
        prediction = model.predict(X_new_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative price
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

print("\nSTEP 12: Prediction Function Created")
print("-" * 40)

# =============================================================================
# PART 13: EXAMPLE PREDICTIONS
# =============================================================================

print("\n1STEP 13: Example Predictions")
print("-" * 40)

# Example predictions
example_houses = [
    {
        'area': 2000, 'bedrooms': 3, 'bathrooms': 2, 'floors': 2, 
        'year_built': 2000, 'location': 'Suburb', 'condition': 'Good', 'garage': 'Yes'
    },
    {
        'area': 2500, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2, 
        'year_built': 2010, 'location': 'Downtown', 'condition': 'Excellent', 'garage': 'Yes'
    },
    {
        'area': 1500, 'bedrooms': 2, 'bathrooms': 1, 'floors': 1, 
        'year_built': 1980, 'location': 'Rural', 'condition': 'Fair', 'garage': 'No'
    }
]

print("Sample House Price Predictions:")
print("=" * 50)

for i, house in enumerate(example_houses, 1):
    prediction = predict_house_price(**house, model_name=best_model_name)
    
    print(f"\n🏠 House {i}:")
    print(f"  Area: {house['area']} sq ft")
    print(f"  Bedrooms: {house['bedrooms']}, Bathrooms: {house['bathrooms']}")
    print(f"  Floors: {house['floors']}, Built: {house['year_built']}")
    print(f"  Location: {house['location']}, Condition: {house['condition']}")
    print(f"  Garage: {house['garage']}")
    
    if prediction:
        print(f"  💰 Predicted Price: ${prediction:,.0f}")
    else:
        print(f"  ❌ Prediction failed")

# =============================================================================
# PART 14: MODEL COMPARISON SUMMARY
# =============================================================================

print(f"\nSTEP 14: Final Model Comparison Summary")
print("=" * 60)

print("Model Performance Ranking (by Test R²):")
results_df_sorted = results_df.sort_values('Test_R2', ascending=False)

for i, (idx, row) in enumerate(results_df_sorted.iterrows(), 1):
    print(f"{i}. {row['Model']}")
    print(f"   R²: {row['Test_R2']:.4f} | MAE: ${row['Test_MAE']:,.0f} | RMSE: ${row['Test_RMSE']:,.0f}")

# Overfitting analysis
print(f"\nOverfitting Analysis:")
for i, (idx, row) in enumerate(results_df_sorted.iterrows(), 1):
    overfitting = row['Train_R2'] - row['Test_R2']
    status = "⚠️ High" if overfitting > 0.1 else "✅ Low" if overfitting < 0.05 else "⚡ Moderate"
    print(f"{i}. {row['Model']}: {status} overfitting ({overfitting:.4f})")

# =============================================================================
# PART 15: INTERACTIVE PREDICTION INTERFACE
# =============================================================================

def interactive_prediction():
    """
    Interactive interface for house price prediction
    """
    print(f"\n🎮 STEP 15: Interactive House Price Prediction")
    print("=" * 50)
    print("Enter house details to get a price prediction:")
    print("(Press Enter to use default values shown in brackets)")
    
    try:
        # Get user input
        area = input("Area in sq ft [2000]: ").strip()
        area = int(area) if area else 2000
        
        bedrooms = input("Number of bedrooms [3]: ").strip()
        bedrooms = int(bedrooms) if bedrooms else 3
        
        bathrooms = input("Number of bathrooms [2]: ").strip()
        bathrooms = int(bathrooms) if bathrooms else 2
        
        floors = input("Number of floors [2]: ").strip()
        floors = int(floors) if floors else 2
        
        year_built = input("Year built [2000]: ").strip()
        year_built = int(year_built) if year_built else 2000
        
        print("\nLocation options: Downtown, Suburb, Rural")
        location = input("Location [Suburb]: ").strip()
        location = location if location in ['Downtown', 'Suburb', 'Rural'] else 'Suburb'
        
        print("Condition options: Poor, Fair, Good, Excellent")
        condition = input("Condition [Good]: ").strip()
        condition = condition if condition in ['Poor', 'Fair', 'Good', 'Excellent'] else 'Good'
        
        print("Garage options: Yes, No")
        garage = input("Garage [Yes]: ").strip()
        garage = garage if garage in ['Yes', 'No'] else 'Yes'
        
        # Make prediction with all available models
        print(f"\nPredictions from all models:")
        print("-" * 40)
        
        predictions = {}
        for model_name in results.keys():
            pred = predict_house_price(
                area=area, bedrooms=bedrooms, bathrooms=bathrooms, floors=floors,
                year_built=year_built, location=location, condition=condition, 
                garage=garage, model_name=model_name
            )
            if pred:
                predictions[model_name] = pred
                print(f"{model_name}: ${pred:,.0f}")
        
        if predictions:
            avg_prediction = np.mean(list(predictions.values()))
            std_prediction = np.std(list(predictions.values()))
            print(f"\nEnsemble Statistics:")
            print(f"Average Prediction: ${avg_prediction:,.0f}")
            print(f"Standard Deviation: ${std_prediction:,.0f}")
            print(f"Prediction Range: ${avg_prediction - std_prediction:,.0f} - ${avg_prediction + std_prediction:,.0f}")
            
            print(f"\n🏆 Best Model Prediction ({best_model_name}): ${predictions[best_model_name]:,.0f}")
        
        return predictions
        
    except KeyboardInterrupt:
        print("\n👋 Prediction cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Error in interactive prediction: {str(e)}")
        return None

# Run interactive prediction
choice = int(input("\n \nEnter Interactive Prediction(0 for YES, 1 for NO):"))
if (choice == 0):
    interactive_predictions = interactive_prediction()

# =============================================================================
# PART 16: SUMMARY
# =============================================================================

print("=" * 60)


print(f"\nDataset Statistics:")
print(f"   - Total samples: {len(df):,}")
print(f"   - Features used: {len(feature_columns)}")
print(f"   - Train/Test split: {len(X_train)}/{len(X_test)}")

print(f"\nModels Trained: {len(models)}")
for model_name in models.keys():
    print(f"   - {model_name}")

print(f"\n🏆Best Performing Model:")
print(f"   - Model: {best_model_name}")
print(f"   - R² Score: {best_r2:.4f}")
print(f"   - Mean Absolute Error: ${results[best_model_name]['test_mae']:,.0f}")

if 'Random Forest' in results:
    top_feature = feature_importance.iloc[0]
    print(f"\n🎯 Most Important Feature:")
    print(f"   - {top_feature['Feature']} (Importance: {top_feature['Importance']:.3f})")

print(f"\nSaved Artifacts:")
saved_files = [
    f"   - Models: {len(models)} trained models",
    f"   - Preprocessing: imputer, scaler, encoders",
    f"   - Results: performance metrics and summaries",
    f"   - Features: feature names and importance"
]
for file_desc in saved_files:
    print(file_desc)

print(f"\nPrediction Capabilities:")
print(f"   - Interactive prediction function available")
print(f"   - Supports all {len(models)} trained models")
print(f"   - Handles categorical encoding automatically")
print(f"   - Includes ensemble prediction statistics")

print(f"\nKey Insights:")
if 'Price' in correlation_matrix.columns:
    top_corr_features = correlation_matrix['Price'].abs().sort_values(ascending=False)[1:4]
    print(f"   - Top price predictors:")
    for feature, corr in top_corr_features.items():
        print(f"     * {feature}: {corr:.3f} correlation")

# Calculate price statistics
price_stats = df['Price'].describe()
print(f"   - Price range: ${price_stats['min']:,.0f} - ${price_stats['max']:,.0f}")
print(f"   - Average price: ${price_stats['mean']:,.0f}")
print(f"   - Median price: ${price_stats['50%']:,.0f}")


print("=" * 60)
