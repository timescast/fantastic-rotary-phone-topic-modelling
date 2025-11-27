
#  Topic Modeling on StackOverflow (10% Sample) Using BERTopic

This project performs **topic modeling** on a 10% sampled StackOverflow dataset using **BERTopic**, **Sentence Transformers**, **UMAP**, and **HDBSCAN**.
It includes preprocessing, embedding generation, topic extraction, visualization, and predicting topics for new documents.

---

##  **Overview**

The workflow includes:

1. **Load StackOverflow subset from Google Drive**
2. **Weight title more lightly and body more heavily**
3. **Convert weighted text into dense embeddings (MiniLM-L6-v2)**
4. **Run BERTopic with custom UMAP/HDBSCAN**
5. **Visualize topics and diagnose clusters**
6. **Handle outliers & update topics**
7. **Predict topics for new unseen documents**
8. **Save outputs and topic info**

This allows efficient topic discovery from large QA datasets.

---

##  **Project Steps**

### **1. Load Data**

```python
docs = pd.read_csv('/content/drive/MyDrive/subset_data.csv')
```

Extracts `Title`, `Body`, and `Tag`.

---

### **2. Weight Titles and Bodies**

To emphasize the problem description more than the title:

```python
TITLE_WEIGHT = 1
BODY_WEIGHT = 2
weighted_documents = ...
```

---

### **3. Generate Dense Embeddings**

Using **all-MiniLM-L6-v2**:

```python
embeddings = embeding_model.encode(weighted_documents)
np.save('/content/drive/MyDrive/embeddings.npy', embeddings)
```

---

### **4. Build BERTopic Pipeline**

Custom UMAP + HDBSCAN pipeline:

```python
umap_model = UMAP(...)
hdbscan_model = HDBSCAN(...)
vectorizer_model = CountVectorizer(...)
representation_model = KeyBERTInspired()
ctfidf_model = ClassTfidfTransformer(...)
```

---

### **5. Fit BERTopic**

```python
topic_model = BERTopic(...)
topics, probs = topic_model.fit_transform(weighted_documents, embeddings)
```

---

### **6. Save/Load the Model**

```python
topic_model.save("/content/drive/MyDrive/models/model00")
loaded_model = BERTopic.load("/content/drive/MyDrive/models")
```

---

### **7. Visualization**

Includes:

* **Topic Map**
* **Hierarchical Tree**
* **Topic Heatmap**
* **Bar Charts**

```python
topic_model.visualize_topics()
topic_model.visualize_heatmap()
topic_model.visualize_barchart()
```

---

### **8. Outlier Detection & Reduction**

Re-run BERTopic with modified parameters and adjust clusters:

```python
new_topics = topic_model.reduce_outliers(...)
topic_model.update_topics(weighted_documents, topics=new_topics)
```

---

### **9. Topic Info Export**

```python
freq = topic_model.get_topic_info()
freq.to_csv("/content/drive/MyDrive/topic_info.csv")
```

---

### **10. Predict Topics for New Documents**

Embedding + similarity-based topic prediction:

```python
predictions = np.argmax(sim_matrix, axis=1)
```

And BERTopic’s native prediction:

```python
new_topic, new_prob = topic_model.transform(new_docs)
```

Outputs predicted topic and confidence score.

---

##  **Outputs**

* `embeddings.npy` — Dense vector representations
* BERTopic model (saved as safetensors)
* Topic frequency table
* Interactive HTML visualizations
* Predictions for new unseen posts

---

##  **Technologies Used**

| Component                    | Details          |
| ---------------------------- | ---------------- |
| **Embeddings**               | all-MiniLM-L6-v2 |
| **Topic Modeling**           | BERTopic         |
| **Dimensionality Reduction** | UMAP             |
| **Clustering**               | HDBSCAN          |
| **Vectorization**            | CountVectorizer  |
| **Representation**           | KeyBERT-inspired |
| **Storage**                  | Google Drive     |

---

##  **How to Run**

Install dependencies:

```bash
pip install bertopic sentence-transformers umap-learn hdbscan
```

Run each section in Colab as in the notebook.

---

##  Future Enhancements

* Automatic topic merging
* Apply dynamic topic modeling (over time analysis)
---

 
