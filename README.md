# FITS: Modeling Time Series with 10k parameters

This is the forecasting part of the FITS. Run the scripts in 'scripts/FITS' to get the results. We will further update the final scripts soon. 

## 🚨 Important Update: 2023-12-25 🎄

We've identified a significant bug in our code, originally found in Informer (AAAI 2021 Best Paper), thanks to an anonymous researcher (who will be credited later). This issue has implications for a broad spectrum of research on time series forecasting, including but not limited to:

- PatchTST (ICLR 2023) - [Link to affected code](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/data_provider/data_factory.py)
- TimesNet (ICLR 2023) - [Link to affected code](https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py) (Note: Not impacted due to specific batch size setting during testing)
- DLinear (AAAI 2022 reported version) - [Link to affected code](https://github.com/cure-lab/LTSF-Linear/commit/6fe4c28ff36b4228792f2bbe513e807577e4a57e)
- Informer (AAAI 2021 Best Paper) - [Link to affected code](https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py)
- Autoformer (NIPS 2021 reported version) - [Link to affected code](https://github.com/thuml/Autoformer/commit/d9100709b04e3e8361170794eba4f47b1afb217f)
- Fedformer (ICML 2022) - [Link to affected code](https://github.com/MAZiqing/FEDformer/blob/master/data_provider/data_factory.py)
- FiLM (ICLR 2023) - [Link to affected code](https://github.com/tianzhou2011/FiLM/blob/main/data_provider/data_factory.py)
- iTransformer (ICLR 2024 score: 8886) - [Link to affected code](https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py)

Efforts are underway to correct this bug, and we will update our Arxiv submission and this repository with the revised results. A bug fix method will also be released to assist the community in addressing this issue in their work.

### Description of the Bug:

The bug stems from an incorrect implementation in the data loader. Specifically, the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data, particularly with large batch sizes, leading to unfair model comparisons.

### Solution:

To fix this issue in codebases using LSTF-Linear's architecture:

1. In [data_factory.py](./data_provider/data_factory.py) within the data_provider folder (usually on line 19), change:

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    ```

    To:

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = False #True
        batch_size = args.batch_size
        freq = args.freq
    ```

2. In your experiment script (e.g., [./exp/exp_main.py](./exp/exp_main_F.py)), modify the following (around line 290):

    From:
    ```python
    preds = np.array(preds)
    trues = np.array(trues)
    inputx = np.array(inputx) # some times there is not this line, it does not matter
    ```

    To:
    ```python
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0) # if there is not that line, ignore this
    ```

    If you do not do this, it will generate an error during testing because of the dimension 0 (batch_size) is not aligned. Maybe this is why everyone is dropping the last batch. But concatenate them on the 0 axis (batch_size) can solve this problem. 

3. Run the officially provided scripts!

## Result Update

| Model     | ETTh1-96  | ETTh1-192 | ETTh1-336 | ETTh1-720 | ETTh2-96  | ETTh2-192 | ETTh2-336 | ETTh2-720 | ETTm1-96  | ETTm1-192 | ETTm1-336 | ETTm1-720 | ETTm2-96  | ETTm2-192 | ETTm2-336 | ETTm2-720 |
| --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| PatchTST  | 0.385     | *0.413*   | *0.44*    | *0.456*   | *0.274*   | *0.338*   | *0.367*   | *0.391*   | **0.292** | **0.33**  | **0.365** | *0.419*   | **0.163** | *0.219*   | *0.276*   | *0.368*   |
| Dlinear   | 0.384     | 0.443     | 0.446     | 0.504     | 0.282     | 0.350     | 0.414     | 0.588     | *0.301*   | *0.335*   | 0.371     | 0.426     | 0.171     | 0.237     | 0.294     | 0.426     |
| FedFormer | **0.375** | 0.427     | 0.459     | 0.484     | 0.340     | 0.433     | 0.508     | 0.480     | 0.362     | 0.393     | 0.442     | 0.483     | 0.189     | 0.256     | 0.326     | 0.437     |
| TimesNet  | 0.384     | 0.436     | 0.491     | 0.521     | 0.340     | 0.402     | 0.452     | 0.462     | 0.338     | 0.374     | 0.410     | 0.478     | 0.187     | 0.249     | 0.321     | 0.408     |
| FITS      | **0.375** | **0.407** | **0.43**  | **0.427** | **0.271** | **0.331** | **0.354** | **0.378** | 0.309     | 0.338     | *0.366*   | **0.414** | **0.163** | **0.217** | **0.268** | **0.349** |
| IMP       | 0.000     | 0.006     | 0.010     | 0.029     | 0.003     | 0.007     | 0.013     | 0.013     | -0.017    | -0.008    | -0.001    | 0.005     | 0.000     | 0.002     | 0.008     | 0.019     |

| Model     | Weather-96 | Weather-192 | Weather-336 | Weather-720 | Electricity-96 | Electricity-192 | Electricity-336 | Electricity-720 | Traffic-96 | Traffic-192 | Traffic-336 | Traffic-720 |
| --------- | ---------- | ----------- | ----------- | ----------- | -------------- | --------------- | --------------- | --------------- | ---------- | ----------- | ----------- | ----------- |
| PatchTST  | *0.151*    | *0.195*     | *0.249*     | *0.321*     | **0.129**      | **0.149**       | *0.166*         | 0.210           | **0.366**  | **0.388**   | **0.398**   | *0.457*     |
| Dlinear   | 0.174      | 0.217       | 0.262       | 0.332       | 0.140          | 0.153           | 0.169           | **0.204**       | 0.413      | 0.423       | 0.437       | 0.466       |
| Fedformer | 0.246      | 0.292       | 0.378       | 0.447       | 0.188          | 0.197           | 0.212           | 0.244           | 0.573      | 0.611       | 0.621       | 0.630       |
| TimesNet  | 0.172      | 0.219       | 0.280       | 0.365       | 0.168          | 0.184           | 0.198           | 0.220           | 0.593      | 0.617       | 0.629       | 0.640       |
| **FITS**  | **0.144**  | **0.188**   | **0.238**   | **0.308**   | *0.135*        | **0.149**       | **0.165**       | **0.204**       | *0.385*    | *0.397*     | *0.411*     | **0.449**   |
| IMP       | 0.007      | 0.007       | 0.011       | 0.013       | -0.006         | 0.000           | 0.001           | 0.000           | -0.019     | -0.009      | -0.013      | 0.008       |

## Analysis

The discovered bug predominantly impacts results on smaller datasets like ETTh1 and ETTh2. Interestingly, for other datasets, certain models, such as PatchTST on ETTm1, demonstrate enhanced performance.

### Replication

- We have uploaded the training logs for community review. Additionally, we've provided logs for other baseline models. It's important to note that these logs were generated using their respective official codebases, not the versions in this repository.

- We will update the training scripts of FITS very soon. 

- For fairness, we have conducted baseline runs using freshly cloned codebases with the original hyperparameters. (Note: Avoid using versions from this repository.) TimesNet, which is unaffected by this issue, was not re-run and is mentioned here only for reference.

- We encourage the community to apply the provided bug fix and re-conduct their experiments.

(A minor note: The only change we made in hyperparameters was reducing the learning rate for DLinear on ETTh2 from 0.05 to 0.005, resulting in improved outcomes.)

(A word of caution: Training PatchTST, particularly on datasets like traffic and electricity, can be extremely time-consuming.)


## Notice

- ~~**FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time.~~
- Please run the scripts for ETT datasets with _fin.

## Update
- We add a notebook for interpretability. We analyze FITS on synthetic datasets to show its capability of modeling sinusodial waves. 
- We add a model Real_FITS which use two linear layer to simulate the complex multiplication. This model can achieve the same result of FITS. Real_FITS can be used on devices that do not support complex number calculation (e.g. RTX4090). 
- We add a onnx implementation of FITS with the architecture of Real_FITS. ONNX is an open format built to represent machine learning models. It can be directly deploy on embedded system devices such as STM32. As far as we know, there is compatability issue on Cube AI with onnx opset17. 
- **All the training scripts are updated!**
- Files for anomaly detection are uploaded! Please check the instruction [here](./AD/runAD.md)
- ⚠ **We find a long standing bug in our code which may affect a wide range of research work. Please check the Important Notice section for more information.** We have been actively fixing this bug and rerun all our experiments as well as the baseline models we compared with. 