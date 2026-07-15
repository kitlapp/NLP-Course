# Natural Language Processing Project

## Dataset Description

The dataset focuses on drug ratings and contains a free-text user review column, along with three rating dimensions: Satisfaction, Effectiveness, and Ease of Use.

## Files

1. `pr_01_data_collection_and_preprocessing.ipynb`
   Jupyter notebook containing the data collection, exploratory analysis, preprocessing, and feature engineering steps.

2. The above notebook produces three final datasets, which are shared through Google Drive due to their size:

   https://drive.google.com/drive/folders/1spDsOxCQ2BARxbTrWTqTPRZEXoxvlDtl?usp=drive_link

   a. `df_analytics_final.parquet`
   Dataset containing valid observations that do not provide meaningful textual information for NLP tasks, preserved for potential future analytical work.

   b. `df_lem_final.parquet`
   NLP-ready dataset with lemmatized reviews, suitable for classical NLP techniques such as Bag of Words, TF-IDF, and traditional machine learning models.

   c. `df_non_lem_final.parquet`
   NLP-ready dataset with non-lemmatized reviews, suitable for modern NLP approaches, including deep learning and transformer-based models.


3. `pr_02_feature_engineering_embeddings.ipynb`
   Jupyter notebook containing the Feature Engineering stage, including TF-IDF and Word2Vec embeddings, word similarity search, PCA and t-SNE visualizations, and a comparison of the embedding techniques.

   The notebook also produces the following trained embedding artifacts, which are shared through Google Drive due to their size:

   https://drive.google.com/drive/folders/<YOUR_DRIVE_LINK>

   a. `tfidf_matrix.npz`
   Sparse TF-IDF document-term matrix generated from the lemmatized review corpus.

   b. `tfidf_vectorizer.joblib`
   Trained TF-IDF vectorizer containing the learned vocabulary and IDF weights.

   c. `word2vec_model.model`
   Trained Word2Vec model containing dense word embeddings learned from the review corpus.

4. `pr_03_unsupervised_learning.ipynb`
   Jupyter notebook containing the Unsupervised Learning stage, where LDA and NMF topic modeling are applied to identify latent topics within the drug review corpus. The notebook includes topic visualization, interpretation, and a comparison of both topic modeling approaches.
