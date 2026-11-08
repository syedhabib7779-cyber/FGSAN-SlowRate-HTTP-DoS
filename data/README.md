# Data

The datasets are public but are **not redistributed** in this repository.

## CICIDS2017 (main experiments)
1. Download *MachineLearningCSV* from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Take `Wednesday-workingHours.pcap_ISCX.csv` (contains BENIGN, DoS slowloris, DoS Slowhttptest).
3. On Colab, place it under `MyDrive/cicids2017/` (Cell 3 of `notebooks/FGSAN_main.ipynb` searches that folder;
   change `SEARCH_ROOT` to use another location).

## CSE-CIC-IDS2018 (external validation, `experiments/M6_cell35_external_ids2018.py`)
The cell downloads `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` and
`Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` from the public AWS bucket
`s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/` (no credentials needed).
Dataset page: https://www.unb.ca/cic/datasets/ids-2018.html

Please cite the dataset authors when using these data:
I. Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and
intrusion traffic characterization," in Proc. ICISSP, 2018, pp. 108–116.
