# spatiotemporal-transformer

## Overview

This is the code for the paper "Spatiotemporal Transformer for 2D Time Series Sequence Forecasting". The code is mainly built upon Kedro and MLflow for experiment management. All models are implemented using PyTorch. 

## Requirements

Make sure you have pipenv version ```2020.8.13``` and up installed. Follow the instruction [here](https://docs.python-guide.org/dev/virtualenvs/). The requirements below will be installed automatically if you follow the installation guide below.
```
kedro==0.16.6
black==v19.10b0
flake8>=3.7.9, <4.0
einops==0.3.0
ipython~=7.0
isort>=4.3.21, <5.0
jupyter~=1.0
jupyter_client>=5.1, < 7.0
jupyterlab==2.2.9
kedro==0.16.6
kedro-mlflow==0.4.0
kedro-viz==3.6.0
linformer==0.2.0
mlflow==1.11.0
nbstripout==0.3.3
netCDF4==1.5.4
pyro-ppl==1.5.1
pytest-cov~=2.5
pytest-mock>=1.7.1, <2.0
pytest~=5.0
seaborn==0.11.0
scikit-learn==0.23.2
tensorboard==2.4.0
torch==1.7.0
torchvision==0.8.1
torchaudio==0.7.0
tqdm==4.51.0
wheel==0.32.2
```

## Installing

```
pipenv install      # Make and install environment
pipenv shell        # Run the environment
kedro install       # Install all the dependencies
pip install pytorch-warmup  # Install warmup learning module for PyTorch
```

## Datasets
The dataset for Moving MNIST and CMAP can be obtained here.
```
http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
ftp://ftp.cdc.noaa.gov/Datasets/cmap/enh/precip.pentad.mean.nc
```
Download them and put the files inside ```/data/01_raw/```.  
Note that for Moving MNIST dataset, you will need to convert the npy file into pkl file first. This can be done by running the script ```/notebooks/moving_mnist.py```. The resulting file will be inside ```/data/01_raw/``` too.
```
cd notebooks
python moving_mnist.py
```

## Running experiments
The experiments can be run by executing the following scripts while in the pipenv environment.
```
# Run Spatiotemporal Transformer on the Moving MNIST dataset
run_exp_spatrans_moving_mnist.sh      
# Run Spatiotemporal Transformer Quantile Regression on the Moving MNIST dataset      
run_exp_spatrans_probs_moving_mnist.sh

# Run Spatiotemporal Transformer on the CMAP dataset  
run_exp_spatrans_cmap.sh                   
# Run Spatiotemporal Transformer Quantile Regression on the CMAP dataset 
run_exp_spatrans_probs_cmap.sh

# Run baseline models on the Moving MNIST dataset
run_exp_baselines_moving_mnist.sh           
# Run baseline quantile regression models on the Moving MNIST dataset
run_exp_baselines_probs_moving_mnist.sh

# Run baseline models on the CMAP dataset
run_exp_baselines_cmap.sh                 
# Run baseline quantile regression models on the CMAP dataset  
run_exp_baselines_probs_cmap.sh             
```
You can change the parameters of the model by going inside the script and change the ```generic_model_params``` to your liking. To run the models on CPU instead of GPU, change the ```use_cuda``` parameter to ```false```, although it will likely be very slow.


## Seeing Experiment Results
Run
```
mlflow ui
```
and go to 
```
http://127.0.0.1:5000
```
to see the MLflow experiment page. You can check all of your experiment results here. More about [MLflow](https://mlflow.org/).


