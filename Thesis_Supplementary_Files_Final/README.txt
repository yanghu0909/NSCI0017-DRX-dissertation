Supplementary files for the thesis computational workflow

This folder accompanies the MSc dissertation and contains the data, notebooks, scripts and numerical outputs required to inspect the machine-learning workflow.

Main workflow
- Featurisation.ipynb: dataset cleaning, fixed formula-level train/validation/test assignment, CBFV generation and feature filtering.
- Training.ipynb: synthesis-condition features, scaling, comparison of nine classifiers, Extra Trees evaluation and candidate prediction.
- run_corrected_training_pipeline.py: reproducible model-comparison and evaluation script.
- make_extratrees_12_sample_predictions.py: Extra Trees predictions for the 12 experimental Ni-Mn-Ti candidates at 900 degrees C for 4 h using the sol-gel method.

Data
- drx dataset.csv: cleaned literature dataset (173 records; 115 unique formulae).
- data_splits/split_manifest.json: exact formula membership of the original train/validation/test split.
- data_splits/drx_train.csv: training split (128 records; 80 unique formulae).
- data_splits/drx_val.csv: validation split (31 records; 23 unique formulae).
- data_splits/drx_test.csv: independent test split (14 records; 12 unique formulae).
- X_train_filtered.csv, X_val_filtered.csv and X_test_filtered.csv: final 106 composition-descriptor matrices.
- low_variance_removed.csv: 92 descriptors removed by zero-variance filtering.
- high_correlation_removed.csv: 66 descriptors removed using an absolute correlation threshold of 0.98.

Final outputs
- model_comparison_classic_models.csv: metrics for the nine compared classifiers.
- best_model_summary.json: final Extra Trees settings and summary metrics.
- classification_report_extra_trees_validation.csv and classification_report_extra_trees_test.csv: class-specific evaluation results.
- confusion_matrix_extra_trees_validation.csv and confusion_matrix_extra_trees_test.csv: validation and independent-test confusion matrices.
- feature_importance_extra_trees.csv: impurity-based Extra Trees feature-importance values.
- predictions_for_12_experimental_samples_extra_trees_with_conditions.csv: predicted labels and probabilities for the 12 experimental candidates.

Feature and preprocessing notes
- CBFV generates 264 composition descriptors; filtering retains 106 descriptors.
- Three synthesis-condition features are added: Method, temperature and time, giving 109 model input features.
- Method is a single integer-coded categorical feature: 0 = Hydrothermal, 1 = Mechanochemical, 2 = Molten-salt, 3 = Sol-gel, 4 = Solid-state. This is not one-hot encoding.
- Atmosphere is excluded from the final model input.
- StandardScaler is fitted only on the training set and then applied unchanged to validation, test and experimental-candidate inputs.

Reproducibility notes
- The exact original composition-level split is stored in data_splits/split_manifest.json. The notebook reuses this manifest rather than drawing a new random split.
- Model estimators use random_state=42 where supported.
- Class labels: 0 = unsuccessful; 1 = successful.
- Run the notebooks with this folder as the working directory.

Software
- Python 3
- pandas
- numpy
- scikit-learn
- CBFV
- jupyter
