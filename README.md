# Heart Disease Clustering

Unsupervised segmentation of the Heart Disease UCI dataset using **K-Means** and **AGNES (Ward linkage)**, following a workflow inspired by **CRISP-DM**.

## Project objective

Explore whether patients can be grouped according to their explanatory clinical and demographic variables, without using the target label to build the clusters.

> **Important:** `target` is excluded from exploratory analysis, preprocessing, PCA, model training, and the selection of the number of clusters. It is used only after clustering for external validation.

## Dataset

- 303 observations
- 13 explanatory variables
- 1 target variable, reserved for post-hoc external validation
- No missing values in the explanatory variables
- One duplicate row detected on explanatory variables and retained for this analysis

The dataset file is not included in this repository. Obtain it from an authorized source and place it in the project directory as `heart.csv`.

## Methodology

1. **Data understanding:** inspect the explanatory variables, descriptive statistics, missing values, and duplicate rows.
2. **Exploratory analysis:** examine distributions, boxplots, and correlations using explanatory variables only.
3. **Preprocessing:** apply IQR clipping to the five continuous variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
4. **Scaling:** standardize the 13 explanatory variables using `StandardScaler`.
5. **Dimensionality reduction:** use PCA for exploratory visualization.
6. **Clustering:** compare K-Means and AGNES with Ward linkage for `k=2` and `k=3`.
7. **External validation:** examine the relationship between the resulting clusters and `target` only after clustering.

## Main results

### Internal comparison

| Algorithm | k | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---:|---:|---:|---:|
| K-Means | 2 | 0.1660 | 2.1978 | 55.2674 |
| AGNES-Ward | 2 | 0.1439 | 2.3699 | 39.9137 |
| K-Means | 3 | 0.1149 | 2.4908 | 41.9598 |
| AGNES-Ward | 3 | 0.0959 | 2.5331 | 33.7105 |

Among the four tested configurations, **K-Means with k=2** obtains the strongest internal validation scores. **K-Means with k=3** is retained as a complementary segmentation to describe three more detailed profiles.

### PCA

The first two principal components explain **33.09%** of the total variance. Therefore, 2D PCA plots provide only a partial view of the data structure and should not be treated as proof of cluster quality.

### Post-hoc external validation of K-Means k=3

- Chi-square: 121.4714 (2 degrees of freedom)
- p-value: 4.19578 × 10⁻²⁷
- Adjusted Rand Index (ARI): 0.2550
- Normalized Mutual Information (NMI): 0.2497

These results indicate an association between the clusters and `target`, but only limited agreement between the two partitions. They are not used to select `k` or train the clustering model.

## Repository structure

```text
Heart_Disease_Clustering/
├── heart_disease_clustering.py
├── heart.csv                         # local dataset; not committed
├── Rapport_Heart_Disease_Clustering_final.pdf
├── outputs/                          # generated tables and figures
└── README.md
```

If you store the report in a subfolder, update the report path above accordingly (for example, `reports/Rapport_Heart_Disease_Clustering_final.pdf`).

## Requirements

Python 3.9+ is recommended.

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Run the project

Place `heart.csv` next to the Python script, then run:

```bash
python heart_disease_clustering.py --data heart.csv
```

The script saves generated tables and visualizations in the `outputs/` directory.

## Interpretation and limitations

- The internal validation scores are relatively modest; cluster separation is limited.
- Numerical encoding of categorical variables can introduce artificial distances when Euclidean-based methods are used.
- K-Means and Ward are sensitive to feature scaling and cluster geometry.
- IQR clipping reduces the influence of extreme values but may also reduce genuine variation.
- The duplicate row was retained and may affect the clustering if it represents a duplicated record.
- The `thal` codes must be interpreted according to the exact dataset version and its data dictionary.
- The clusters are exploratory statistical segments, **not medical diagnoses or clinical risk scores**.

## Tools

Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn

## Reproducibility note

The reported metrics correspond to the execution results documented for this project. Re-run the script with the same dataset version and environment to regenerate the outputs. Cluster labels are arbitrary identifiers and may be permuted between runs.
