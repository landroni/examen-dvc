##activate venv
cd accidents/examen-dvc/
source ../Template_MLOps_accidents/env/bin/activate



##create modified dataset
python ./src/data/make_dataset.py
# python ./src/data/make_dataset.py ./data/raw ./data/processed  

##build features
python ./src/features/build_features.py

##train model
python ./src/models/train_model.py

##evaluate model
python ./src/models/evaluate_model.py

##predict model
python ./src/models/predict_model.py




