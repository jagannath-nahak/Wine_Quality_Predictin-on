import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


MODEL_FILE = "wine_model.pkl"
PIPELINE_FILE = "wine_pipeline.pkl"


def build_pipeline(num_columns):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_columns)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):

    # 1. load dataset
    wine = pd.read_csv("WineQT.csv")

    # drop id column if exists
    if "Id" in wine.columns:
        wine = wine.drop("Id", axis=1)

    # 2. split train test
    train_set, test_set = train_test_split(
        wine, test_size=0.2, random_state=42
    )

    # save test for inference
    test_set.drop("quality", axis=1).to_csv("input.csv", index=False)

    # 3. separate labels
    wine_labels = train_set["quality"].copy()
    wine_features = train_set.drop("quality", axis=1)

    # 4. numerical columns
    num_columns = wine_features.select_dtypes(include=[np.number]).columns

    # 5. build pipeline
    pipeline = build_pipeline(num_columns)

    wine_prepared = pipeline.fit_transform(wine_features)

    # 6. train model
    model = RandomForestRegressor(random_state=42)
    model.fit(wine_prepared, wine_labels)

    # save
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Wine model trained successfully!")

else:

    # inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    transformed_input = pipeline.transform(input_data)

    predictions = model.predict(transformed_input)

    input_data["quality"] = predictions

    input_data.to_csv("output.csv", index=False)

    print("Wine quality prediction complete!")