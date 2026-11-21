"""
Heart Disease UCI — segmentation non supervisée (CRISP-DM)
Exécution :
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    python heart_disease_clustering.py --data heart.csv

Règle de séparation :
- target est isolée dès le chargement.
- target n'intervient JAMAIS dans l'EDA explicative, le clipping,
  le scaling, la PCA, le choix de k ou l'entraînement.
- target est réintroduite uniquement dans la dernière section de validation externe.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import chi2_contingency

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10})

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
EXPECTED_FEATURES = CONTINUOUS_COLS + CATEGORICAL_COLS
K_VALUES = [2, 3]
RANDOM_STATE = 42
N_INIT = 20
OUTPUT_DIR = Path("outputs")


def save_show(fig, filename):
    """Enregistre une figure puis l'affiche."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def compute_metrics(X, labels, algorithm, k, inertia=np.nan):
    """Calcule les métriques internes sur l'espace standardisé complet."""
    return {
        "Algorithme": algorithm,
        "k": k,
        "Silhouette (↑)": silhouette_score(X, labels),
        "Davies-Bouldin (↓)": davies_bouldin_score(X, labels),
        "Calinski-Harabasz (↑)": calinski_harabasz_score(X, labels),
        "Inertie (↓, K-Means)": inertia,
        "Taille minimale": pd.Series(labels).value_counts().min(),
        "Taille maximale": pd.Series(labels).value_counts().max(),
    }


def thal_description(values):
    """
    Les fichiers Heart UCI circulent avec plusieurs codages de thal.
    Ne pas attribuer de sens clinique à un code sans vérifier la version.
    """
    vals = set(pd.Series(values).dropna().astype(int).unique())
    if vals.issubset({1, 2, 3}) and 0 not in vals:
        return {
            1: "normal (codage Kaggle fréquent)",
            2: "défaut fixé (codage Kaggle fréquent)",
            3: "défaut réversible (codage Kaggle fréquent)",
        }
    if vals.issubset({3, 6, 7}):
        return {
            3: "normal (codage UCI original)",
            6: "défaut fixé (codage UCI original)",
            7: "défaut réversible (codage UCI original)",
        }
    return {int(v): "code à documenter selon la source" for v in sorted(vals)}


def main(data_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PHASE 1 — BUSINESS / DATA UNDERSTANDING
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    if "target" not in df.columns:
        raise ValueError("La colonne 'target' est absente du fichier.")
    missing_features = sorted(set(EXPECTED_FEATURES) - set(df.columns))
    if missing_features:
        raise ValueError(
            "Colonnes attendues absentes : "
            + ", ".join(missing_features)
            + ". Vérifiez la version du dataset."
        )

    # Séparation stricte immédiate. target ne sera utilisée qu'à la fin.
    y_external = df["target"].copy()
    X_raw = df.drop(columns=["target"]).copy()

    # On conserve les variables explicatives attendues et leur ordre.
    X_raw = X_raw[EXPECTED_FEATURES].copy()

    print("\n=== PHASE 1 — COMPRÉHENSION DES DONNÉES ===")
    print(f"Observations : {len(X_raw)}")
    print(f"Variables explicatives : {X_raw.shape[1]}")
    print(f"Doublons : {X_raw.duplicated().sum()}")
    print("\nValeurs manquantes par variable :")
    print(X_raw.isna().sum())
    print("\nStatistiques descriptives :")
    print(X_raw.describe().T.round(2))
    print("\nCodes observés pour thal :")
    print(sorted(X_raw["thal"].dropna().unique()))
    print("Dictionnaire indicatif thal :", thal_description(X_raw["thal"]))

    if X_raw.isna().any().any():
        raise ValueError(
            "Des valeurs manquantes existent. Définir une stratégie d'imputation "
            "sur X uniquement avant de poursuivre."
        )

    # Contrôle du type numérique : les algorithmes attendent une matrice numérique.
    non_numeric = X_raw.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(f"Variables non numériques à encoder : {non_numeric}")

    # ------------------------------------------------------------------
    # PHASE 2 — EDA : target explicitement exclue
    # ------------------------------------------------------------------
    print("\n=== PHASE 2 — ANALYSE EXPLORATOIRE (SANS TARGET) ===")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Distribution des variables numériques continues", fontweight="bold")
    for ax, col in zip(axes.flat, CONTINUOUS_COLS):
        sns.histplot(X_raw[col], bins=25, kde=True, ax=ax, color="steelblue")
        ax.axvline(X_raw[col].mean(), color="darkred", linestyle="--",
                   label=f"Moyenne = {X_raw[col].mean():.1f}")
        ax.set_title(col)
        ax.set_xlabel({
            "age": "Âge (années)",
            "trestbps": "Pression au repos (mmHg)",
            "chol": "Cholestérol (mg/dl)",
            "thalach": "Fréquence cardiaque maximale (bpm)",
            "oldpeak": "Dépression ST (unités du dataset)",
        }[col])
        ax.set_ylabel("Nombre de patients")
        ax.legend()
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save_show(fig, "01_distributions_continues.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Distribution des variables catégorielles codées", fontweight="bold")
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        counts = X_raw[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="slateblue")
        ax.set_title(col)
        ax.set_xlabel("Code observé")
        ax.set_ylabel("Nombre de patients")
    fig.tight_layout()
    save_show(fig, "02_distributions_categorielles.png")

    # Corrélation de Pearson calculée sur les codes numériques.
    # Pour les variables catégorielles ordinalisées, elle ne prouve pas une
    # relation linéaire substantielle et doit être interprétée avec prudence.
    corr = X_raw.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Corrélations de Pearson — variables explicatives uniquement")
    fig.tight_layout()
    save_show(fig, "03_matrice_correlation.png")
    corr.to_csv(OUTPUT_DIR / "correlations_features.csv")

    fig, axes = plt.subplots(1, len(CONTINUOUS_COLS), figsize=(16, 4))
    fig.suptitle("Valeurs extrêmes avant clipping IQR", fontweight="bold")
    for ax, col in zip(axes, CONTINUOUS_COLS):
        sns.boxplot(y=X_raw[col], ax=ax, color="lightsteelblue")
        ax.set_title(col)
        ax.set_ylabel("Valeur observée")
    fig.tight_layout()
    save_show(fig, "04_boxplots_avant_clipping.png")

    # ------------------------------------------------------------------
    # PHASE 3 — DATA PREPARATION : clipping IQR sur variables continues
    # ------------------------------------------------------------------
    print("\n=== PHASE 3 — PRÉTRAITEMENT ===")
    X_clean = X_raw.copy()
    iqr_rows = []

    for col in CONTINUOUS_COLS:
        q1 = X_clean[col].quantile(0.25)
        q3 = X_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((X_clean[col] < lower) | (X_clean[col] > upper)).sum())
        X_clean[col] = X_clean[col].clip(lower=lower, upper=upper)
        iqr_rows.append({
            "Variable": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Borne inférieure": lower, "Borne supérieure": upper,
            "Valeurs écrêtées": n_outliers,
        })

    iqr_table = pd.DataFrame(iqr_rows)
    print("\nBornes et valeurs écrêtées par clipping IQR :")
    print(iqr_table.round(2).to_string(index=False))
    iqr_table.to_csv(OUTPUT_DIR / "bornes_iqr.csv", index=False)

    # Les variables catégorielles restent codées en nombres entiers :
    # c'est une limite méthodologique explicite du présent protocole.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

    print("\nVérification StandardScaler (moyennes proches de 0, écarts-types proches de 1) :")
    print(pd.DataFrame({
        "Moyenne": X_scaled_df.mean(),
        "Écart-type": X_scaled_df.std(ddof=0),
    }).round(3))

    # ------------------------------------------------------------------
    # PHASE 4 — PCA : visualisation uniquement
    # ------------------------------------------------------------------
    print("\n=== PHASE 4 — PCA ===")
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    explained = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    pca_table = pd.DataFrame({
        "Composante": [f"PC{i+1}" for i in range(len(explained))],
        "Variance expliquée (%)": explained,
        "Variance cumulée (%)": cumulative,
    })
    print(pca_table.head(5).round(2).to_string(index=False))
    print(f"\nVariance expliquée par PC1 + PC2 : {cumulative[1]:.2f}%")
    pca_table.to_csv(OUTPUT_DIR / "pca_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_comp = np.arange(1, len(explained) + 1)
    ax.bar(x_comp, explained, alpha=0.65, label="Variance individuelle (%)")
    ax.plot(x_comp, cumulative, "o-", color="darkred",
            label="Variance cumulée (%)")
    ax.axhline(80, color="darkgreen", linestyle="--", label="Repère 80 %")
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance expliquée (%)")
    ax.set_title("PCA — variance individuelle et cumulée")
    ax.legend()
    fig.tight_layout()
    save_show(fig, "05_pca_variance.png")

    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    pca_2d_pct = pca_2d.explained_variance_ratio_ * 100

    # ------------------------------------------------------------------
    # PHASE 5 — MODELING : comparer k=2 et k=3, K-Means et AGNES
    # ------------------------------------------------------------------
    print("\n=== PHASE 5 — COMPARAISON DES MODÈLES ===")
    all_labels = {}
    metric_rows = []

    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        km_labels = km.fit_predict(X_scaled)
        all_labels[("K-Means", k)] = km_labels
        metric_rows.append(
            compute_metrics(X_scaled, km_labels, "K-Means", k, km.inertia_)
        )

        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        ag_labels = ag.fit_predict(X_scaled)
        all_labels[("AGNES-Ward", k)] = ag_labels
        metric_rows.append(
            compute_metrics(X_scaled, ag_labels, "AGNES-Ward", k)
        )

    metrics_df = pd.DataFrame(metric_rows)
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.to_csv(OUTPUT_DIR / "comparaison_modeles_k2_k3.csv", index=False)

    # Courbes de sélection K-Means sur k=2..8.
    curve_rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_scaled)
        curve_rows.append({
            "k": k,
            "Inertie": model.inertia_,
            "Silhouette": silhouette_score(X_scaled, labels),
            "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
            "Calinski-Harabasz": calinski_harabasz_score(X_scaled, labels),
        })
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUTPUT_DIR / "courbes_selection_kmeans.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Sélection de k — K-Means (espace standardisé à 13 variables)",
                 fontweight="bold")
    axes[0].plot(curve_df["k"], curve_df["Inertie"], "o-", color="navy")
    axes[0].set_title("Méthode du coude")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Inertie (somme des carrés intra-clusters)")
    axes[1].plot(curve_df["k"], curve_df["Silhouette"], "o-", color="darkgreen")
    axes[1].set_title("Score de Silhouette")
    axes[1].set_xlabel("Nombre de clusters (k)")
    axes[1].set_ylabel("Silhouette (plus élevé = mieux)")
    axes[2].plot(curve_df["k"], curve_df["Davies-Bouldin"], "o-", color="darkred")
    axes[2].set_title("Indice Davies-Bouldin")
    axes[2].set_xlabel("Nombre de clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin (plus faible = mieux)")
    for ax in axes:
        ax.set_xticks(range(2, 9))
    fig.tight_layout()
    save_show(fig, "06_selection_nombre_clusters.png")

    # Choix analytique provisoire : les deux k sont conservés pour examen.
    # La solution finale par défaut est k=3 uniquement pour reproduire le scénario
    # du rapport initial ; elle doit être confirmée après lecture des métriques.
    selected_k = 3
    selected_algorithm = "K-Means"
    selected_labels = all_labels[(selected_algorithm, selected_k)]
    print(f"\nSolution affichée pour les profils : {selected_algorithm}, k={selected_k}")
    print("Attention : le choix doit être justifié à partir du tableau comparatif, "
          "pas imposé par cette valeur par défaut.")

    # Projection PCA 2D : illustration uniquement, pas l'espace d'entraînement.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Comparaison des partitions — projection PCA 2D "
                 "(visualisation uniquement)", fontweight="bold")
    plot_specs = [
        ("K-Means", 2, axes[0, 0]), ("K-Means", 3, axes[0, 1]),
        ("AGNES-Ward", 2, axes[1, 0]), ("AGNES-Ward", 3, axes[1, 1]),
    ]
    for algorithm, k, ax in plot_specs:
        labels = all_labels[(algorithm, k)]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                             cmap="tab10", s=30, alpha=0.75)
        ax.set_title(f"{algorithm} — k={k}")
        ax.set_xlabel(f"PC1 ({pca_2d_pct[0]:.2f} %)")
        ax.set_ylabel(f"PC2 ({pca_2d_pct[1]:.2f} %)")
        ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    save_show(fig, "07_comparaison_partitions_pca.png")

    # Dendrogramme sur un échantillon reproductible (lecture qualitative seulement).
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(80, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
    Z = linkage(X_scaled[sample_idx], method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title(f"Dendrogramme AGNES-Ward — échantillon de {sample_size} patients")
    ax.set_xlabel("Observations de l'échantillon (identifiants masqués)")
    ax.set_ylabel("Distance de fusion (Ward)")
    fig.tight_layout()
    save_show(fig, "08_dendrogramme_agnes.png")

    # ------------------------------------------------------------------
    # PHASE 6 — EVALUATION ET PROFILS : valeurs brutes après clipping
    # ------------------------------------------------------------------
    print("\n=== PHASE 6 — PROFILS DES CLUSTERS ===")
    profile_data = X_clean.copy()
    profile_data["cluster"] = selected_labels
    profile_means = profile_data.groupby("cluster")[CONTINUOUS_COLS].mean()
    profile_medians = profile_data.groupby("cluster")[CONTINUOUS_COLS].median()
    cluster_sizes = profile_data["cluster"].value_counts().sort_index()

    print("\nMoyennes réelles des variables continues (unités originales, après clipping) :")
    print(profile_means.round(2).to_string())
    print("\nMédianes :")
    print(profile_medians.round(2).to_string())
    print("\nEffectifs :")
    print(cluster_sizes.to_string())

    profile_means.to_csv(OUTPUT_DIR / "profils_moyens_bruts.csv")
    profile_medians.to_csv(OUTPUT_DIR / "profils_medians_bruts.csv")
    cluster_sizes.rename("Effectif").to_csv(OUTPUT_DIR / "effectifs_clusters.csv")

    # Graphique des moyennes en unités réelles : une figure par variable.
    fig, axes = plt.subplots(1, len(CONTINUOUS_COLS), figsize=(17, 4))
    fig.suptitle(f"Moyennes réelles par cluster — {selected_algorithm}, k={selected_k}",
                 fontweight="bold")
    units = {
        "age": "années", "trestbps": "mmHg", "chol": "mg/dl",
        "thalach": "bpm", "oldpeak": "unités du dataset",
    }
    for ax, col in zip(axes, CONTINUOUS_COLS):
        ax.bar(profile_means.index.astype(str), profile_means[col], color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"Moyenne ({units[col]})")
        for i, value in enumerate(profile_means[col]):
            ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_show(fig, "09_moyennes_reelles_par_cluster.png")

    # Profil min-max pour comparer les formes, sans remplacer le tableau brut.
    denom = profile_means.max() - profile_means.min()
    profile_minmax = (profile_means - profile_means.min()) / denom.replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id, row in profile_minmax.iterrows():
        ax.plot(CONTINUOUS_COLS, row.values, marker="o", linewidth=2,
                label=f"Cluster {cluster_id}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Moyenne min-max (0 à 1, calculée entre clusters)")
    ax.set_title("Comparaison relative des profils — échelle min-max")
    ax.legend(title="Groupes")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_show(fig, "10_profils_minmax.png")

    # Répartition des variables catégorielles par cluster : fréquences relatives.
    for col in CATEGORICAL_COLS:
        ctab = pd.crosstab(profile_data["cluster"], X_clean[col], normalize="index")
        ctab.to_csv(OUTPUT_DIR / f"distribution_{col}_par_cluster.csv")

    # ------------------------------------------------------------------
    # PHASE 7 — VALIDATION EXTERNE A POSTERIORI (TARGET)
    # ------------------------------------------------------------------
    # Jusqu'ici, y_external n'a été consultée ni pour le prétraitement,
    # ni pour la PCA, ni pour le choix de k, ni pour les modèles.
    print("\n=== PHASE 7 — VALIDATION EXTERNE A POSTERIORI ===")
    external_table = pd.crosstab(
        pd.Series(selected_labels, name="Cluster"),
        y_external.rename("target"),
        margins=True,
    )
    external_percent = pd.crosstab(
        pd.Series(selected_labels, name="Cluster"),
        y_external.rename("target"),
        normalize="index",
    ) * 100

    print("\nTableau croisé cluster x target (effectifs) :")
    print(external_table)
    print("\nRépartition de target au sein de chaque cluster (%) :")
    print(external_percent.round(1))
    external_table.to_csv(OUTPUT_DIR / "validation_externe_effectifs.csv")
    external_percent.to_csv(OUTPUT_DIR / "validation_externe_pourcentages.csv")

    # Mesures descriptives d'association, calculées après gel des clusters.
    contingency = pd.crosstab(selected_labels, y_external)
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        ari = adjusted_rand_score(y_external, selected_labels)
        nmi = normalized_mutual_info_score(y_external, selected_labels)
        print(f"\nChi-deux d'indépendance : chi2={chi2:.4f}, ddl={dof}, p={p_value:.6g}")
        print(f"Adjusted Rand Index (validation externe) : {ari:.4f}")
        print(f"Normalized Mutual Information (validation externe) : {nmi:.4f}")
        print("Ces mesures décrivent l'association avec target ; elles ne servent "
              "pas à choisir k ni à réentraîner le clustering.")
        pd.DataFrame([{
            "Chi2": chi2, "ddl": dof, "p_value": p_value,
            "ARI_externe": ari, "NMI_externe": nmi,
        }]).to_csv(OUTPUT_DIR / "metriques_validation_externe.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(external_percent, annot=True, fmt=".1f", cmap="Blues",
                cbar_kws={"label": "Patients du cluster (%)"}, ax=ax)
    ax.set_title("Validation externe — distribution de target par cluster")
    ax.set_xlabel("target (étiquette diagnostique du dataset)")
    ax.set_ylabel("Cluster non supervisé")
    fig.tight_layout()
    save_show(fig, "11_validation_externe_target.png")

    # ------------------------------------------------------------------
    # EXPORT FINAL
    # ------------------------------------------------------------------
    results = X_clean.copy()
    results["cluster_kmeans_k2"] = all_labels[("K-Means", 2)]
    results["cluster_kmeans_k3"] = all_labels[("K-Means", 3)]
    results["cluster_agnes_k2"] = all_labels[("AGNES-Ward", 2)]
    results["cluster_agnes_k3"] = all_labels[("AGNES-Ward", 3)]
    # target est jointe uniquement dans ce fichier d'audit final, après clustering.
    results["target_validation_externe"] = y_external.values
    results.to_csv(OUTPUT_DIR / "patients_clusters_validation.csv", index=False)

    print("\n=== FIN ===")
    print(f"Tous les tableaux et graphiques sont enregistrés dans : {OUTPUT_DIR.resolve()}")
    print("La variable target apparaît uniquement dans les exports et analyses "
          "de validation externe réalisés après le clustering.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="heart.csv",
                        help="Chemin vers le fichier CSV Heart Disease UCI")
    args = parser.parse_args()
    main(args.data)
