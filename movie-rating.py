import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
data = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
data['Year'] = data['Year'].str.extract(r'(\d{4})')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

data['Duration'] = data['Duration'].str.replace("min", "").str.strip()
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')

data['Votes'] = pd.to_numeric(data['Votes'].str.replace(",", ""), errors='coerce')

data = data.dropna(subset=['Rating'])
X = data[['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = data['Rating']

categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
numeric_features = ['Year', 'Duration', 'Votes']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', max_categories=50))
        ]), categorical_features),
        
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]), numeric_features)
    ])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("RMSE:", rmse)
print("R² Score:", r2)
def show_top_features(model, categorical_features, numeric_features, top_n=15):
    cat_features = model.named_steps['preprocessor'] \
        .transformers_[0][1].named_steps['encoder'].get_feature_names_out(categorical_features)
    num_features = numeric_features
    all_features = np.concatenate([cat_features, num_features])
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    feature_importance['Readable Feature'] = feature_importance['Feature'].apply(
        lambda x: x.split("_", 1)[1] if "_" in x else x
    )

    print("\nTop Important Features (with names):")
    print(feature_importance[['Readable Feature', 'Importance']])
    plt.figure(figsize=(10,6))
    plt.barh(feature_importance['Readable Feature'], feature_importance['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Important Features for Movie Rating Prediction")
    plt.show()
show_top_features(model, categorical_features, numeric_features, top_n=15)
