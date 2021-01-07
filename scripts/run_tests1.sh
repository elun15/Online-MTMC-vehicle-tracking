# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

source activate env_elg_37


# Code to train ResNet50 Model + 2048 classifier batch 100 only training features




cd ..
chmod +x main_cnn_features.py
python main_cnn_features.py --ConfigPath "config/config1.yaml"
cd scripts/

cd ..
chmod +x main_cnn_features.py
python main_cnn_features.py --ConfigPath "config/config2.yaml"
cd scripts/

cd ..
chmod +x main_cnn_features.py
python main_cnn_features.py --ConfigPath "config/config3.yaml"
cd scripts/








