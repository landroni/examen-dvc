import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import sys
print(os.path.dirname('src/data/'))
sys.path.append('src/data/')
from check_structure import check_existing_file


def build_features(processed_path = "data/processed", 
                   scaled_path = "data/scaled"):
    ##FIXME could split this into further functions (import, scale, etc.)
    ##load data prepared by make_dataset
    print("Using processed_path:", processed_path)
    path_X_train = f"{processed_path}/X_train.csv"
    path_X_test = f"{processed_path}/X_test.csv"

    for i_path in [path_X_train, path_X_test]:
        if not os.path.exists(i_path):
            raise FileNotFoundError(f"Fichier introuvable: {i_path}")
    
    X_train = pd.read_csv(path_X_train)
    X_test = pd.read_csv(path_X_test)

    print(f"X_train, X_test: {[i.shape for i in (X_train, X_test)]}")

    ##standardize data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

    print(X_train_scaled.describe().round(2))
    print("Standardization done")

    # Save dataframes to their respective output file paths
    os.makedirs(scaled_path, exist_ok=True)
    save_dataframes(X_train_scaled, X_test_scaled, scaled_path)

    print(f"Saved X_train_scaled, X_test_scaled to {scaled_path}")

    return None

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)



if __name__ == "__main__":
    build_features()