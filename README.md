# TemBERTure

### THERMAL CATEGORY CLASSIFICATION TASK: TemBERTure_CLS 

TemBERTureCLS is a sequence-based classifier leveraging ProtBert and adopting an adapter-based approach, following the methodology proposed by Pfeiffer et al. (2020) and Houlsby et al. (2019). This classifier is fine-tuned to predict the thermal category (thermophilic, non-thermophilic) based on sequence data.

### MELTING TEMPERATURE PREDICTION TASK : TemBERTure_Tm

TemBERTure_Tm serves as a regression model designed to inference protein melting temperatures (Tm) derived solely from protein sequences.
This repository provides implementations and weights for both tasks, allowing users to leverage these models for various protein-related predictive tasks.

## How to use TemBERTure 
```
model = TemBERTure(adapter_path=path, device='cuda:6')
```
## Dataset availability

Datasets that were used to train, validate, and test TemStaPro are available in Zenodo.

## Interpretation of the results

