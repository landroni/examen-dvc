#################
##activate venv
cd accidents/examen-dvc/
source ../Template_MLOps_accidents/env/bin/activate  ##where exam-DVC venv resides



#################
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



#################
##DVC
##create local dvc repo
dvc init
dvc config core.analytics false  ##disable sending analytics (opt)

##set up DagsHub as DVC remote
dvc remote add remote_dagshub s3://dvc
dvc remote modify remote_dagshub endpointurl https://dagshub.com/landroni/examen-dvc.s3

dvc remote modify remote_dagshub --local access_key_id 5b6faf5dddd0c1b5a532f91eb75ebcbff200413d
dvc remote modify remote_dagshub --local secret_access_key 5b6faf5dddd0c1b5a532f91eb75ebcbff200413d

dvc remote default remote_dagshub
dvc push


#################
##Add files to DVC
##rm indiv data dirs from git tracking
git rm -r --cached 'data/raw'
git rm -r --cached 'data/processed'
git rm -r --cached 'data/predictions'

##add dir to DVC
dvc add data/raw
dvc add data/processed
dvc add data/predictions

##add model binary file to DVC
dvc add models/trained_model.joblib

##stage all DVC dirs/files 
git add .
git commit -m "Add /data dirs and /models files to DVC"

##(3)push dvc data to remote folder & git push
dvc push
git push


#################
##DVC : Pipeline




