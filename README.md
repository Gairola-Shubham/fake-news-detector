# fake-news-detector

---

## 📊 Dataset

- Source: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)
- Merged and labeled: `label = 1` for Fake, `label = 0` for Real
- Preprocessed with URL removal, punctuation cleanup, and deduplication

---

## 🧪 Modeling Steps

1. **TF-IDF + Logistic Regression**
   - Bigram features
   - Class-weight balanced
   - ROC AUC and classification report

2. **SBERT + XGBoost**
   - Sentence embeddings via `all-MiniLM-L6-v2`
   - Tree-based classifier with 300 estimators

3. **RoBERTa Fine-Tuning**
   - Hugging Face Trainer API
   - Evaluation on validation and test sets
   - F1 and accuracy metrics

---

## 🔍 Explainability

- **LIME**: Local explanations for TF-IDF predictions
- **SHAP**: Global feature importance for XGBoost

---

## 🖥 Streamlit Demo

Run locally:
```bash
streamlit run app.py
