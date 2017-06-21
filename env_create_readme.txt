conda env create -f requirements/aind-dog-mac.yml
source activate aind-dog
KERAS_BACKEND=tensorflow python -c "from keras import backend"
