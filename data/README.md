# Prepare the dataset

To generate the MC simulation groundtruth, please run
```
python prepare_dataset.py --dataset DATASET_NAME
```

DATASET_NAME could be one of 'citeseer', 'cora_ml', 'ms_academic', 'pubmed'

In the prepare_dataset.py file, some simulation parameters could be altered, e.g., seed size, edge probability, number of samples for each seed size, etc.
