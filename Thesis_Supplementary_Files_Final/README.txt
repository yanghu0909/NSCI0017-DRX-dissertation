Supplementary files for the thesis computational workflow

This archive accompanies the final thesis and contains the data, code and key numerical outputs required to inspect or reproduce the machine-learning workflow.

Main workflow
- Featurisation.ipynb: dataset cleaning, formula-grouped splitting, CBFV generation and feature filtering.
- Training.ipynb: preprocessing, comparison of nine classifiers, Extra Trees selection and evaluation.
- run_corrected_training_pipeline.py: reproducible final model-comparison and evaluation script.
- make_extratrees_12_sample_predictions.py: final Extra Trees predictions for the 12 experimental compositions at 900 degrees C for 4 h using the sol-gel method.

Data
- drx dataset.csv: complete cleaned literature dataset (173 entries).
- data_splits/drx_train.csv: training split (128 entries).
- data_splits/drx_val.csv: validation split (31 entries).
- data_splits/drx_test.csv: independent test split (14 entries).
- X_train_filtered.csv, X_val_filtered.csv and X_test_filtered.csv: final 106 composition-descriptor matrices.
- low_variance_removed.csv: 92 descriptors removed by zero-variance filtering.
- high_correlation_removed.csv: 66 descriptors removed using the absolute correlation threshold of 0.98.

Final outputs
- model_comparison_classic_models.csv: metrics for the nine compared classifiers.
- best_model_summary.json: final Extra Trees settings and summary metrics.
- classification_report_extra_trees_validation.csv and classification_report_extra_trees_test.csv: class-specific evaluation results.
- confusion_matrix_extra_trees_validation.csv and confusion_matrix_extra_trees_test.csv: validation and independent-test confusion matrices.
- feature_importance_extra_trees.csv: ungrouped impurity-based Extra Trees feature importance values.
- predictions_for_12_experimental_samples_extra_trees_with_conditions.csv: final predicted labels and probabilities for the 12 experimental samples.

Software
- Python 3
- pandas
- numpy
- scikit-learn
- CBFV
- jupyter

Reproducibility notes
- Random seed: 42.
- Class labels: 0 = unsuccessful; 1 = successful.
- Synthesis method is encoded as a single ordinal feature with codes 0-4.
- StandardScaler is fitted only to the training set and then applied unchanged to the validation, test and experimental-candidate inputs.
- The complete source dataset and split files are included; generated figures are already presented in the thesis and are therefore not duplicated here.
