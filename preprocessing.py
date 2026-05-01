import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer 


data = pd.read_csv('updated_main_dataset_3.csv') 
data = data.drop(columns=['Unnamed: 0'], errors='ignore')  

numeric_cols = ['Latitude', 'Longitude', 'Elevation', 'RSRP', 'RSRQ', 'SNR']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

initial_rows = len(data)
data = data.dropna(subset=['Latitude', 'Longitude', 'Elevation'])
print(f"Dropped {initial_rows - len(data)} rows due to missing geospatial data.")
data = data.reset_index(drop=True)

location_cols = ['Latitude', 'Longitude']
agg_dict = {col: 'mean' for col in ['RSRP', 'RSRQ', 'SNR']}
agg_dict['Elevation'] = 'median'  
data = data.groupby(location_cols).agg(agg_dict).reset_index()
print(f"Reduced to {len(data)} unique locations after aggregating duplicates (from {initial_rows} prior rows).")

impute_cols = ['RSRP', 'RSRQ', 'SNR']
missing_summary = {col: data[col].isna().sum() for col in impute_cols}
print(f"Missing values before imputation: {missing_summary}")

feature_cols = ['Latitude', 'Longitude', 'Elevation']
X_features = data[feature_cols]
X_to_impute = data[impute_cols]

scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = imputer.fit_transform(np.column_stack([X_features_scaled, X_to_impute]))

data[impute_cols] = X_imputed[:, len(feature_cols):]

for col in impute_cols:
    data[col] = data[col].fillna(data[col].median())

data = data[(data['RSRP'] >= -120) & (data['RSRP'] <= -40)]  
data = data[(data['RSRQ'] >= -25) & (data['RSRQ'] <= 0)]
data = data[(data['SNR'] >= -20) & (data['SNR'] <= 40)]

data['RSRP'] = data['RSRP'].round(0).astype(int)  
data['RSRQ'] = data['RSRQ'].round(2)  
data['SNR'] = data['SNR'].round(2)    
data['Elevation'] = data['Elevation'].round(1)  

print(f"Missing values after imputation: {data[impute_cols].isna().sum().to_dict()}")
print(f"Final dataset has {len(data)} rows after outlier clipping.")
print(data.head())

data.to_csv('preprocessed_main_dataset_3.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_main_dataset_3.csv'")