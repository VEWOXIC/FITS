# FITS: Modeling Time Series with 10k parameters

This is the forecasting part of the FITS. Run the scripts in 'scripts/FITS' to get the results. We will further update the final scripts soon. 

## ⚠⚠⚠Important Notice⚠⚠⚠ 2023-12-25🎄

An anonymous researcher (we will acknowledge him/her later) finds a long standing bug in our code architecture which can be traced back to Informer (AAAI 2021 Best Paper). This bug affect a wide range of research work which include but not limit to:

- PatchTST (ICLR 2023) (https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/data_provider/data_factory.py )
- ~~TimesNet (ICLR 2023) (https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py)~~ [We later find that Timesnet set the batch_size as 1 during testing. Thus it is not affected by this bug.]
- DLinear (AAAI 2022 reported version) (https://github.com/cure-lab/LTSF-Linear/commit/6fe4c28ff36b4228792f2bbe513e807577e4a57e)
- Informer (AAAI 2021 Best Paper) (https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py)
- Autoformer (NIPS 2021 reported version) (https://github.com/thuml/Autoformer/commit/d9100709b04e3e8361170794eba4f47b1afb217f )
- Fedformer (ICML 2022) (https://github.com/MAZiqing/FEDformer/blob/master/data_provider/data_factory.py )
- FiLM (ICLR 2023) (https://github.com/tianzhou2011/FiLM/blob/main/data_provider/data_factory.py )
- iTransformer (ICLR 2024 score: 8886) (https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py)

We have been actively fixing this bug and rerun all our experiments. We will update the result on the arxiv and this repo. Also we will release a bug fix method for the community to fix their code and re-evaluate their work. 

We and the anonymous friend hope the community can actively fix this long-standing problem together. 

### Bug:

The bug is caused by the incorrect implementation of the data loader. The test dataloader incorrectly use the drop_last=True, which may cause a large portion of test data being especially when the batch size is large. This may result in unfair comparison between different models with different batch size.

### Fix:

For the codebase that uses the code architecture of LSTF-Linear(https://github.com/cure-lab/LTSF-Linear), please follow the following instruction to fix the bug:

1. Modify the [data_factory.py](./data_provider/data_factory.py) in the data_provider folder. (mostly on line 19)
   
   
    from

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    ```

    to

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = False #True
        batch_size = args.batch_size
        freq = args.freq
    ```

2. Modify the experiment script [./exp/exp_main.py](./exp/exp_main_F.py) or any experiment script that the model runs on (on around line 290)

    from
    ```python
    preds = np.array(preds)
    trues = np.array(trues)
    inputx = np.array(inputx) # some times there is not this line, it does not matter
    ```

    to
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
| Dlinear   | 0.384     | 0.443     | 0.446     | 0.504     | 0.282     | 0.35      | 0.414     | 0.588     | *0.301*   | *0.335*   | 0.371     | 0.426     | 0.171     | 0.237     | 0.294     | 0.426     |
| FedFormer | **0.375** | 0.427     | 0.459     | 0.484     | 0.34      | 0.433     | 0.508     | 0.48      | 0.362     | 0.393     | 0.442     | 0.483     | 0.189     | 0.256     | 0.326     | 0.437     |
| TimesNet  | 0.384     | 0.436     | 0.491     | 0.521     | 0.34      | 0.402     | 0.452     | 0.462     | 0.338     | 0.374     | 0.41      | 0.478     | 0.187     | 0.249     | 0.321     | 0.408     |
| FITS      | **0.375** | **0.407** | **0.43**  | **0.427** | **0.271** | **0.331** | **0.354** | **0.378** | 0.309     | 0.338     | *0.366*   | **0.414** | **0.163** | **0.217** | **0.268** | **0.349** |
| IMP       | 0         | 0.006     | 0.01      | 0.031     | 0.003     | 0.007     | 0.013     | 0.013     | -0.017    | -0.008    | -0.001    | 0.005     | 0         | 0.002     | 0.008     | 0.019     |

| Model     | Weather-96 | Weather-192 | Weather-336 | Weather-720 | Electricity-96 | Electricity-192 | Electricity-336 | Electricity-720 | Traffic-96 | Traffic-192 | Traffic-336 | Traffic-720 |
| --------- | ---------- | ----------- | ----------- | ----------- | -------------- | --------------- | --------------- | --------------- | ---------- | ----------- | ----------- | ----------- |
| PatchTST  | *0.151*    | *0.195*     | *0.249*     | *0.321*     | **0.129**      | **0.149**       | *0.166*         | 0.21            | **0.366**  | **0.388**   | **0.398**   | *0.457*     |
| Dlinear   | 0.174      | 0.217       | 0.262       | 0.332       | 0.14           | 0.153           | 0.169           | **0.204**       | 0.413      | 0.423       | 0.437       | 0.466       |
| Fedformer | 0.246      | 0.292       | 0.378       | 0.447       | 0.188          | 0.197           | 0.212           | 0.244           | 0.573      | 0.611       | 0.621       | 0.63        |
| TimesNet  | 0.172      | 0.219       | 0.28        | 0.365       | 0.168          | 0.184           | 0.198           | 0.22            | 0.593      | 0.617       | 0.629       | 0.64        |
| **FITS**  | **0.144**  | **0.188**   | **0.238**   | **0.308**   | *0.135*        | **0.149**       | **0.165**       | **0.204**       | *0.385*    | *0.397*     | *0.411*     | **0.449**   |
| IMP       | 0.007      | 0.007       | 0.011       | 0.013       | -0.006         | 0               | 0.001           | 0.006           | -0.019     | -0.009      | -0.013      | 0.008       |

## Analyze

This bug mainly affect the result on smaller dataset such as ETTh1 and ETTh2. On other datasets, the result even shows a better performance in some cases (e.g. PatchTST on ETTm1 and FITS on ).

### Reproduce

We have update the training script for FITS. Also the training log is uploaded for community to check. We also upload the log for other baselines. Note that these logs are get with the corresponding official codebases instead of the ones in this repo.

We run other baselines with freshly cloned codebase and the provided hyperparameters. (DO NOT USE THE ONES IN THIS REPO FOR FAIRNESS) (We did not run TimesNet since it is not suffer from this issue, and we put it here for reference) 

(We change the learning rate of DLinear on ETTh2 from 0.05 to 0.005 for a better result. That is the only hyper-parameter we change. )

(Kindly remind: the training of PatchTST can takes forever, especially on traffic and electricity dataset. )

## Notice

- ~~**FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time.~~
- Please run the scripts for ETT datasets with _fin.

## Update
- We add a notebook for interpretability. We analyze FITS on synthetic datasets to show its capability of modeling sinusodial waves. 
- We add a model Real_FITS which use two linear layer to simulate the complex multiplication. This model can achieve the same result of FITS. Real_FITS can be used on devices that do not support complex number calculation (e.g. RTX4090). 
- We add a onnx implementation of FITS with the architecture of Real_FITS. ONNX is an open format built to represent machine learning models. It can be directly deploy on embedded system devices such as STM32. As far as we know, there is compatability issue on Cube AI with onnx opset17. 
- **All the training scripts are updated!**
- Files for anomaly detection are uploaded! Please check the instruction [here](./AD/runAD.md)
- ⚠ **We find a long standing bug in our code which may affect a wide range of research work. Please check the [Notice](#notice) section for more information.** We have actively fixing this bug and rerun all our experiments as well as the baseline models we compared with. 