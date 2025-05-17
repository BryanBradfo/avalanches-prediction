import rampwf as rw

import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Prediction of the avalanche risk'
_target_column_name = 'risque_maxi'
_prediction_label_names = [1, 2, 3, 4, 5]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='accuracy', precision=4),
]

# Features to convert from object to int
features_to_convert = ['precipitation_neige_veille_altitude',
                       'precipitation_neige_veille_epaisseur', 'mer_de_nuages',
                       'limite_pluie_neige', 'isotherme_0',
                       'isotherme_moins_10', 'altitude_vent_1',
                       'altitude_vent_2', 'vitesse_vent_altitude_1',
                       'vitesse_vent_altitude_2']

# Features to keep after feature engineering
# (features that have a correlation with the target greater than 0.1)
features_to_keep = ['temps', 'direction_vent_altitude_1',
                    'direction_vent_altitude_2', 'evolurisque1',
                    'evolurisque2', 'altitude',
                    'precipitation_neige_veille_epaisseur',
                    'limite_pluie_neige', 'isotherme_0',
                    'isotherme_moins_10', 'vitesse_vent_altitude_1',
                    'is_pluie', 'risque_maxi']


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def reshape_massif_data(df):
    """
    Reshapes the dataframe to keep all columns while ensuring
    each day has 3 rows (one for 00:00, one for 06:00, and one for 12:00).
    """

    # Identify core variable names (without time prefixes)
    core_vars = [
        "temps", "mer_de_nuages", "limite_pluie_neige", "isotherme_0",
        "isotherme_moins_10", "altitude_vent_1", "altitude_vent_2",
        "direction_vent_altitude_1", "vitesse_vent_altitude_1",
        "direction_vent_altitude_2", "vitesse_vent_altitude_2"
    ]

    # Keep non-time-dependent columns
    other_columns = [
        "date", "massif", "evolurisque1", "evolurisque2", "altitude",
        "risque1", "risque2", "risque_maxi",
        "precipitation_neige_veille_altitude",
        "precipitation_neige_veille_epaisseur"
    ]

    reshaped_data = []

    # Process each time
    for time in ["00", "06", "12"]:
        # Select the time-specific columns
        selected_cols = [f"{time}_{var}" for var in core_vars]

        # Extract the relevant columns
        df_subset = df[other_columns + selected_cols].copy()

        # Rename time-prefixed columns to remove the time prefix
        df_subset.columns = other_columns + core_vars

        # Set the correct time for each row
        df_subset["date"] = df_subset["date"].dt.normalize() + \
            pd.to_timedelta(f"{time}:00:00")

        # Append transformed dataframe
        reshaped_data.append(df_subset)

    # Concatenate the three time-based DataFrames
    return pd.concat(reshaped_data).sort_values(by=["massif", "date"]
                                                ).reset_index(drop=True)


def convert_object_to_int(df, columns_to_convert):
    """
    Converts the columns in columns_to_convert from object to int.
    """
    df = df.copy()
    for col in columns_to_convert:
        column_data_numeric = pd.to_numeric(df[col], errors='coerce')
        if col == 'precipitation_neige_veille_epaisseur':
            df['is_pluie'] = (df[col] == "Pluie").astype(int)
        if col == 'mer_de_nuages':
            df['no_mer_de_nuages'] = (df[col] == "Non").astype(int)
            column_data_numeric = df[col].replace({
                "Non": 0,
                "Absence de données": -1
            }).infer_objects(copy=False).astype(int)
        column_data_numeric = column_data_numeric.ffill()
        if column_data_numeric.isnull().sum() > 0:
            column_data_numeric = column_data_numeric.bfill()
        df[col] = column_data_numeric
    return df


def load_data(path='.', file='X_train.csv'):
    X_df = pd.read_csv(path / file)

    if X_df['date'].dtype != 'datetime64[ns]':
        X_df['date'] = pd.to_datetime(X_df['date'])
    # Reshape the data
    X_df = reshape_massif_data(X_df)

    # set the date as index
    X_df.set_index('date', inplace=True)

    # Convert object columns to int
    X_df = convert_object_to_int(X_df, features_to_convert)

    # Feature engineering
    if file.name != "bera_clean.csv":
        # numeric_data = X_df.select_dtypes(include=['number'])
        # corr_matrix = numeric_data.corr()
        # features = corr_matrix[_target_column_name][
        #     abs(corr_matrix[_target_column_name]) > 0.1].index.tolist()
        # features = [f for f in features if f not in ["risque1", "risque2"]]
        X_df = X_df[features_to_keep]

    y = X_df[_target_column_name]
    X_df = X_df.drop(columns=[_target_column_name])

    return X_df, y


# READ DATA
def get_train_data(massif='ANDORRE', path='.'):
    folder = Path(path) / "data" / massif
    file = folder / 'X_train.csv'
    return load_data(path, file)


def get_test_data(massif='ANDORRE', path='.'):
    folder = Path(path) / "data" / massif
    file = folder / 'X_test.csv'
    return load_data(path, file)


def get_data(path='.'):
    file = Path(path) / "data" / "bera_clean.csv"
    return load_data(path, file)
