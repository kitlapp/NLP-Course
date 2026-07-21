# Natural Language Processing Project

## Dataset Description

The dataset focuses on drug ratings and contains a free-text user review column, along with three rating dimensions: 
Satisfaction, Effectiveness, and Ease of Use.

## Reproduce the Project

To execute the project's notebooks, a venv setup is not needed because Google Colab VMs handle the required environment 
successfully. In some cases, such as the `gensim` library, manual installation is required. These installations are 
included in the execution flow, inside code cells, and do not require any manual intervention from the executor.

Regarding file reading, proper path handling is achieved by leveraging `kagglehub` mechanics and the `pathlib` module. 
The raw data is read directly from Kaggle without any manual path configuration. This approach was tested during the 
migration of the notebook from local Jupyter Lab to Google Colab, where the raw dataset was successfully retrieved and 
loaded. This selection improves the project's reproducibility, as updates to the dataset page in Kaggle can be 
reflected in the notebooks when the dataset is retrieved again.

For dataset exchange between the notebooks of this project, Google Drive sharing with appropriate permissions is 
selected as the most efficient method. The output uploads to Google Drive are taking place automatically. The same holds 
for the input reading in the beginning of each notebook. Normally, all notebooks have to be executed smoothly, without
manual intervention, nor any path handling.

**ASSUMPTIONS:** The executor must have access to the shared Google Drive folder. Assuming that the Google Drive folder 
is shared with user "X" using the email account "X", the user must open and execute the Colab notebooks using the same 
Google Account associated with email "X". Also, the shared dataset folder should be available inside the user's Google 
Drive root directory, i.e., My Drive. If the shared data folder cannot be seen in the Google Drive root, find it under 
the "Shared with me" section, right-click it → Organize → Add shortcut → All locations → My Drive → Add.

## Files

1. **`pr_01_data_collection_and_preprocessing.ipynb`**  
   Notebook containing the data collection, exploratory analysis, preprocessing, and feature engineering steps.


2. The above notebook produces three final datasets, which are shared through Google Drive due to their size:

   a. **`df_analytics_final.parquet`**
   
   Dataset containing valid observations that do not provide meaningful textual information for NLP tasks, 
   preserved for potential future analytical work.

   b. **`df_lem_final.parquet`**  
   NLP-ready dataset with lemmatized reviews, suitable for classical NLP techniques such as Bag of Words, TF-IDF, and 
   traditional machine learning models.

   c. **`df_non_lem_final.parquet`**  
   NLP-ready dataset with non-lemmatized reviews, suitable for modern NLP approaches, including deep learning and 
   transformer-based models.


3. **`pr_02_feature_engineering_embeddings.ipynb`**  
   Notebook containing text feature engineering, including TF-IDF representation, Word2Vec embedding training, and 
   related feature analysis/visualizations.


4. The above notebook produces the following feature engineering outputs, which are shared through Google Drive:

   a. **`tfidf_matrix.npz`**  
   Sparse matrix containing the TF-IDF representation of the processed reviews. This representation is suitable for 
   traditional NLP techniques and machine learning models.

   b. **`tfidf_vectorizer.joblib`**  
   Saved TF-IDF vectorizer containing the vocabulary and transformation parameters required to reproduce the TF-IDF 
   representation on the same or new text data.

   c. **`word2vec_model.model`**  
   Trained Word2Vec model containing dense word embeddings learned from the review corpus. These embeddings can be 
   used for NLP tasks requiring semantic word representations.  


5. **`pr_03_unsupervised_learning.ipynb`**  
   Notebook containing unsupervised learning techniques for topic discovery, including topic modeling approaches such 
   as Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF), along with topic analysis, 
   visualization, and interpretation of the extracted topics.  


6. **`pr_04_supervised_learning_xai.ipynb`**  
   Notebook containing supervised learning approaches for drug review classification. Different machine learning and 
   deep learning models are applied, including Logistic Regression, Support Vector Machine (SVM), Word2Vec-based 
   classification, LSTM, and transformer-based models. The notebook focuses on training and evaluating the models, 
   comparing their performance, and discussing the results obtained from the different approaches.

   
7. **`pr_042_RAG.ipynb`**  
    Notebook implementing a Retrieval-Augmented Generation (RAG) pipeline for conversational question-answering using the drug reviews dataset. The notebook transforms the processed reviews into searchable vector representations using the pretrained Hugging Face embedding model **`BAAI/bge-small-en-v1.5`** and stores them in a FAISS vector database for efficient similarity-based retrieval. 
    A pretrained causal language model **`Qwen/Qwen3.5-0.8B`** is used to generate responses based on the most relevant retrieved review chunks. Prompt engineering techniques are applied to guide the model usage of review metadata, numerical ratings, and specific fields such as side effects, effectiveness, satisfaction, and ease of use. 
    Finally, the notebook evaluates the RAG system using RAGAS metrics to measure response quality, including faithfulness and answer relevancy. The generated evaluation dataset and results are stored for reproducibility and future analysis.
    The above notebook produces the following RAG pipeline outputs, which are shared through the Google Drive folder **`nlp_project_data`**:  


8. The above notebook produces the following feature engineering outputs, which are shared through Google Drive:
    
   a. **`review_embeddings_<model_name>.npy`**  
   Saved dense vector representations of the review chunks generated by the embedding model. These embeddings are used for similarity search during the retrieval step.

   b. **`review_index_<model_name>.faiss`**  
   FAISS vector index containing the review embeddings. It enables efficient retrieval of the most relevant review chunks for each user query.

   c. **`my_rag_eval_dataset`**  
   Directory containing the evaluation dataset created for RAGAS. It includes the test questions, generated responses, and retrieved contexts used for evaluating the RAG pipeline.

   d. **`ragas_evaluation_results.xlsx`**  
   Excel file containing the RAGAS evaluation results, including the calculated metrics for the generated responses.


