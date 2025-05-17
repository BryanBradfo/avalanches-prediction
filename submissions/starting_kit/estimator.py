
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num_features = ['evolurisque1', 'evolurisque2', 'altitude',
                'precipitation_neige_veille_epaisseur', 'limite_pluie_neige',
                'isotherme_0', 'isotherme_moins_10', 'vitesse_vent_altitude_1',
                'is_pluie']
cat_features = ['temps', 'direction_vent_altitude_1',
                'direction_vent_altitude_2']


def get_estimator():

    # Preprocessing Pipelines
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Define Model Pipeline
    model_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000)
    )

    return model_pipeline
