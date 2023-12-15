# Run FITS for anomaly detection

We develop the FITS for AD based on the code base of [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer/tree/main). We only upload the necessary file here. 

- Clone the anomaly transformer code base (you can skip if already have one). 
```bash
git clone https://github.com/thuml/Anomaly-Transformer.git
```
- Navigate to the working directory. 
```bash
cd Anomaly-Transformer
```
- Move the files in this folder to the working directory and replace the existing one.
- 
dataloader.py -> Anomaly-Transformer/data_provider/dataloader.py

FITS.py -> Anomaly-Transformer/model/FITS.py

main.py -> Anomaly-Transformer/main.py

solver_recon.py -> Anomaly-Transformer/solver_recon.py

- Run the scripts provided in this folder to conduct experiments.

