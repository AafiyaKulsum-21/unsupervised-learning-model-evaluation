# Week 3 Unsupervised Learning and Model Evaluation

This project demonstrates clustering, dimensionality reduction, and supervised model evaluation on a reproducible synthetic customer dataset. The data contain 320 rows and four numeric behavior features. `known_segment` records the generating segment for a limited clustering quality check; it is not included in the clustering input. `high_value` is a separate binary target for logistic regression.

## Run

Requires Python 3.10 or newer. The analysis uses only the standard library.

```powershell
python week3_analysis.py
```

The command writes `customer_segments.csv` and `week3_results.json` in the project folder. A fixed random seed makes the data split and reported results reproducible.

To rebuild the Word report, install `python-docx` in your document-authoring environment and run `python build_report.py` after the analysis. The generated report is `Week_3_Model_Evaluation_Report.docx`.

## Methods

- Standardize features before K-Means and Ward hierarchical clustering.
- Compare K-Means cluster counts from 2 to 6 using inertia, silhouette score, and adjusted Rand index against the known synthetic segment labels.
- Use principal component analysis to summarize four standardized features in two components.
- Split off a stratified 25% test set. Tune logistic regression regularization with stratified five-fold cross-validation on the remaining training data.
- Report the final confusion matrix, accuracy, precision, recall, F1, specificity, and ROC-AUC on the untouched test set.

The data are synthetic and designed for instruction. Results demonstrate a workflow and should not be interpreted as findings about actual customers.
