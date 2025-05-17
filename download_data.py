from pathlib import Path

import pandas as pd

DATA_PATH = Path('data')


if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Load the data
    print('Loading the data...', end='', flush=True)
    file_path = DATA_PATH / "bera.csv"
    X_df = pd.read_csv(file_path)

    # Clean the data
    X_df = X_df.drop_duplicates()
    X_df = X_df.drop(columns=['commentaire', 'url_telechargement',
                              'loc1', 'loc2'])

    # Preprocess the data
    X_df['risque_maxi'] = pd.to_numeric(X_df['risque_maxi'], errors='coerce')
    X_df = X_df.dropna(subset=['risque_maxi'])
    X_df = X_df[X_df['risque_maxi'] != -1]

    # Fill missing values
    X_df['risque2'] = X_df['risque2'].fillna(X_df['risque1'])

    # Convert date to datetime
    X_df['date'] = pd.to_datetime(X_df['date'])

    # Create the target (ou bien max entre risque1 et risque2 ?)
    y = X_df['risque_maxi']

    # Create a dictionary of dataframes for each massif
    massifs = X_df['massif'].unique()
    df_massifs = {massif:
                  X_df[X_df['massif'] == massif] for massif in massifs}

    # Process data separately for each massif
    for massif, df in df_massifs.items():
        # Create folder for the massif
        massif_folder = DATA_PATH / massif
        massif_folder.mkdir(parents=True, exist_ok=True)

        # Fill missing values by bfill
        df['evolurisque1'] = df['evolurisque1'].ffill()
        df['evolurisque2'] = df['evolurisque2'].ffill()
        df['altitude'] = df['altitude'].ffill()

        # Split data
        X_train = df[df['date'].dt.year < 2023]
        X_test = df[df['date'].dt.year >= 2023]

        # Save files inside each massif's folder
        X_train.to_csv(massif_folder / 'X_train.csv', index=False)
        X_test.to_csv(massif_folder / 'X_test.csv', index=False)

    X_df.to_csv(DATA_PATH / 'bera_clean.csv', index=False)

    print('Data splitting and saving completed.')
    print('Each massif has its own folder.')
    print('done')
