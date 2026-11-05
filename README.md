# Data-driven Screening of Li-ion Disordered Rocksalt Cathodes

Supplementary code and data for the UCL MSc Advanced Materials Science dissertation:

**Data-driven Screening and Experimental Validation of Phase Formation in Li-ion Disordered Rocksalt Ni–Mn–Ti Cathodes**

This repository documents a literature-data and machine-learning workflow for screening the formation of disordered rocksalt (DRX) phases in Li-ion cathode materials. A classifier trained on published synthesis records was used to screen 12 Ni–Mn–Ti candidate compositions for experimental validation.

## Dataset

The final curated dataset contains **173 literature records** covering **115 unique composition formulae**. The source CSV contains 9 rows with only `Atmosphere` metadata and no composition or synthesis outcome; these are removed before modelling. The raw `Atmosphere` column is retained as literature metadata but excluded from the final model input.

| Split | Records | Unique formulae |
|---|---:|---:|
| Training | 128 | 80 |
| Validation | 31 | 23 |
| Independent test | 14 | 12 |
| **Total** | **173** | **115** |

The split unit is the unique composition formula. Repeated literature records for the same formula remain in one split. The exact formula membership is stored in `Thesis_Supplementary_Files_Final/data_splits/split_manifest.json`. The notebook reuses this manifest rather than generating a new random split, preserving the split used for the reported results.

The dataset contains 144 successful and 29 unsuccessful records. The class imbalance is considered when interpreting model performance.

## Computational workflow

1. **Data cleaning and labels:** clean literature records and encode the DRX outcome (`1` = successful; `0` = unsuccessful).
2. **Composition featurisation:** generate 264 composition descriptors using `CBFV`.
3. **Feature filtering:** remove zero-variance features and highly correlated features using an absolute Pearson correlation threshold of 0.98. Filtering is determined from the training data. The final composition matrix contains 106 descriptors.
4. **Synthesis conditions:** add synthesis method, temperature (`T`) and time (`t`) to obtain 109 model input features. Atmosphere is excluded from the final model input.
5. **Scaling:** fit `StandardScaler` on the training set only, then apply the same fitted scaler to validation, test and experimental-candidate inputs.
6. **Model comparison:** compare nine classifiers: Dummy Classifier, Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, K-Nearest Neighbours, SVC and LinearSVC.
7. **Evaluation and screening:** select the model by validation Macro F1, evaluate it on the independent test set and predict the 12 experimental candidates under the specified sol-gel conditions (900 °C, 4 h).

### Synthesis-method encoding

Method is represented by one integer-coded categorical feature. **This is not one-hot encoding.**

| Code | Synthesis method |
|---:|---|
| 0 | Hydrothermal |
| 1 | Mechanochemical |
| 2 | Molten-salt |
| 3 | Sol-gel |
| 4 | Solid-state |

Integer coding is the representation used in the dissertation workflow. Since synthesis methods are nominal categories, integer coding can introduce an artificial ordering for algorithms that interpret numerical thresholds or distances. This is a limitation of the current feature representation.

## Model results

Extra Trees was selected using validation Macro F1.

| Metric | Validation | Independent test |
|---|---:|---:|
| Accuracy | 0.806 | 0.857 |
| Balanced accuracy | 0.642 | 0.667 |
| Macro F1 | 0.642 | 0.708 |

The independent test set contains 14 records, including only 3 unsuccessful cases. These metrics should therefore be interpreted cautiously. Accuracy alone is insufficient for this imbalanced dataset; Balanced Accuracy and Macro F1 provide additional information about performance across both classes.

The classifier predicted 10 of the 12 experimental candidates as successful and 2 as unsuccessful. The prediction probabilities and synthesis conditions are recorded in the supplementary CSV. These are screening predictions, not measured phase fractions.

## Repository structure

```text
NSCI0017-DRX-dissertation/
├── README.md
└── Thesis_Supplementary_Files_Final/
    ├── Featurisation.ipynb
    ├── Training.ipynb
    ├── README.txt
    ├── requirements.txt
    ├── drx dataset.csv
    ├── data_splits/
    │   ├── split_manifest.json
    │   ├── drx_train.csv
    │   ├── drx_val.csv
    │   └── drx_test.csv
    ├── run_corrected_training_pipeline.py
    ├── make_extratrees_12_sample_predictions.py
    ├── X_train_filtered.csv / X_val_filtered.csv / X_test_filtered.csv
    ├── model_comparison_classic_models.csv
    ├── best_model_summary.json
    ├── classification_report_*.csv
    ├── confusion_matrix_*.csv
    ├── feature_importance_extra_trees.csv
    └── predictions_for_12_experimental_samples_extra_trees_with_conditions.csv
```

The saved result files can be inspected without rerunning the notebooks. In particular, `model_comparison_classic_models.csv`, `best_model_summary.json` and `predictions_for_12_experimental_samples_extra_trees_with_conditions.csv` provide the classifier comparison, selected-model summary and candidate predictions, respectively.

## Reproducing the workflow

Use Python 3 and install the dependencies:

```bash
cd Thesis_Supplementary_Files_Final
python -m pip install -r requirements.txt
```

Open JupyterLab:

```bash
jupyter lab
```

Run `Featurisation.ipynb` first and `Training.ipynb` second. The notebooks expect `Thesis_Supplementary_Files_Final` to be the working directory. The frozen split manifest preserves the original formula membership. Model estimators use `random_state=42` where supported.

The standalone scripts can also be run from this directory:

```bash
python run_corrected_training_pipeline.py
python make_extratrees_12_sample_predictions.py
```

## Limitations

- The literature dataset is relatively small and class-imbalanced.
- Published literature may under-report unsuccessful synthesis outcomes.
- Synthesis method is integer-coded rather than one-hot encoded.
- The target is a binary DRX formation label; it does not represent quantitative phase fraction or electrochemical performance.
- Experimental screening predictions should be interpreted alongside structural characterisation results.

## Software

- Python 3
- pandas
- NumPy
- scikit-learn
- CBFV
- Jupyter Notebook / JupyterLab
