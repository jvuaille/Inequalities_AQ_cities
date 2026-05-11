# Exploring Inequalities in Urban Air Pollution

- **`PCA_plain.py`**  
  Performs Principal Component Analysis (PCA) across selected variables to derive the main dimensions of inequality in air pollution exposure.

- **`clustering_EU.ipynb`**  
  Jupyter Notebook that clusters European urban areas based on air quality and demographic metrics. Includes data preparation, clustering (e.g. k-means), and visualization of cluster profiles.

- **`indication_hotspots.py`**  
  Identifies “air quality and demographic hotspots” as defined in Vuaille et. al (2026). _Exploring inequalities related to air pollution in European urban areas_.

- Datasets:
    - census2021_AQ_EU28_urban.csv provides the compiled dataset: air quality, census, and degrees of urbanisation
    - AQ_census_2021_EU_clusters_F_stat.csv provides an overview the number of clusters per country and associated pseudo-F statistics
    - AQ_census_2021_EU_PCA_clusters_batch1.csv, AQ_census_2021_EU_PCA_clusters_batch2.csv and AQ_census_2021_EU_PCA_clusters_batch3.csv provide the results from the national clustering analyses in 3 different batches
    - AQ_census_2021_EU_PCA_clusters_batch1_with_indicators, AQ_census_2021_EU_PCA_clusters_batch2_with_indicators and AQ_census_2021_EU_PCA_clusters_batch3_with_indicators add extra features to AQ_census_2021_EU_PCA_clusters_batch datasets to identify hotspots

