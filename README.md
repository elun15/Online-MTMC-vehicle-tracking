## Online-MTMC-vehicle-tracking



# Files and folders
* **./postprocessing** : for post-processing scripts
* **./config** : for configuration .yaml files
* **./misc** : misc and utils scripts
* **./network** : for different network architectures definitions
* **./preprocessing_data** : scripts to process different datasets
* **./scripts** : sh script to run several runnings with different config files
* **./thirdparty** : 3rd party algorithms
* **camera.py** : class for camera transformations 
* **clustering.py** : clustering module
* **colors.py** : indistinguishable colors class
* **dataset.py** : class for dataset transformations
* **display.py** : class for displaying
* **features.py** : feature extractor module
* **main.py** : main file to run 
* **sct.py** : module loading single camera tracking
* **tracking.py**: tracking module
* **env_MTMC.yaml** : Anaconda environment dependencies


Run python main.py --ConfigPath ./config/config.yaml

# Setup
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.7
* Anaconda
* Pycharm

**Anaconda environment**

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
$ conda env create -f env_MTMC.yaml
$ conda activate env_MTMC
```

**Clone repository**

```
$ git clone https://github.com/elun15/Online-MTMC-vehicle-tracking.git
```

# Citation

If you find this code and work useful, please consider citing:




