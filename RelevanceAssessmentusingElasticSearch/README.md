# 🔍 Relevance Assessment using ElasticSearch

This project demonstrates how to assess the **relevance of documents** using **ElasticSearch**.  
It focuses on indexing, searching, and evaluating document relevance with Python.

---

## 📘 Project Overview
- Connected to **ElasticSearch** using Python (`elasticsearch5` library)  
- Loaded and indexed datasets in **JSON format**  
- Used **ElasticSearch queries** to retrieve documents based on relevance  
- Performed reindexing with `helpers.reindex` for dataset updates  
- Evaluated the relevance of search results and improved indexing strategies  

---

## 🛠️ Tech Stack
- **Languages & Libraries**: Python, JSON  
- **ElasticSearch**: Indexing, searching, reindexing  
- **Python Libraries**:  
  - `elasticsearch5` → Connect and interact with ElasticSearch  
  - `json` → Handle data input/output  

---

## ⚙️ Workflow

1. **Data Loading**
   - Load JSON dataset into Python using `json` library  

2. **Connecting to ElasticSearch**
   ```python
   from elasticsearch5 import Elasticsearch
   es = Elasticsearch()
