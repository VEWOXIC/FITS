# FITS: Modeling Time Series with 10k parameters

This is the forecasting part of the FITS. Run the scripts in 'scripts/FITS' to get the results. We will further update the final scripts soon. 

## Notice

- **FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time. 
- Please run the scripts for ETT datasets with _fin.

## Update
- We add a notebook for interpretability. We analyze FITS on synthetic datasets to show its capability of modeling sinusodial waves. 
- We add a model Real_FITS which use two linear layer to simulate the complex multiplication. This model can achieve the same result of FITS. Real_FITS can be used on devices that do not support complex number calculation (e.g. RTX4090). 
- We add a onnx implementation of FITS with the architecture of Real_FITS. ONNX is an open format built to represent machine learning models. It can be directly deploy on embedded system devices such as STM32. As far as we know, there is compatability issue on Cube AI with onnx opset17. 