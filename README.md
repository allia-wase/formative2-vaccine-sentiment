# Formative 2: Sequential Models for Vaccine Sentiment Analysis

Research-informed comparison of sequential modelling approaches for classifying vaccine-related tweets
as negative (-1), neutral (0) or positive (1).
Dataset: [To Vaccinate or Not to Vaccinate (Zindi)](https://zindi.world/competitions/to-vaccinate-or-not-to-vaccinate)

## Approaches
1. TF-IDF + Logistic Regression / Ridge (baseline 1)
2. fastText (baseline 2)
3. BiLSTM + GloVe-Twitter
4. Transformer encoder from scratch
5. Fine-tuned BERTweet / COVID-Twitter-BERT

## Repo layout
```
data/raw/          Train.csv, Test.csv (not committed)
data/processed/    shared train/val/test split (committed)
src/               preprocessing.py, evaluation.py (shared by everyone)
notebooks/         01_eda.ipynb, 02_baselines.ipynb, ...
reports/           figures/, predictions/, results.csv, experiment_log.csv
```

## Team and contributions

### Person 1 – Alliane Umutoniwase: Data & Baselines
- **EDA** (`notebooks/01_eda.ipynb`): class balance, annotator agreement, tweet length, vocabulary, noise (users, URLs, hashtags, emojis), duplicates, negation, and the tweet (`RQMQ0L2A`) that a stray newline splits over two lines.
- **Shared pipeline** (`src/`): cleaning, the fixed stratified train/val/test split, the prediction-output format and the shared experiment log.
- **Baseline 1:** TF-IDF + Logistic Regression / Ridge (experiments B1–B5).
- **Baseline 2:** fastText (experiments F1–F2) (`notebooks/02_baselines.ipynb`).
- **Report:** dataset and EDA section, data preparation, evaluation metrics, and methodology and results for both baselines.

### Person 2 – _Name_: Literature + Models 3 & 4
- _To be filled in: literature review, BiLSTM + GloVe-Twitter, Transformer encoder from scratch, experiments, notebooks and report sections._

### Person 3 – _Name_: Model 5 + Error Analysis + Final Comparison
- _To be filled in: fine-tuned BERTweet / COVID-Twitter-BERT, error analysis, VADER side experiment, final tables and figures, notebooks and report sections._

## Shared rules (everyone)
- Load data with `load_splits()`; never re-split. Split is stratified 70/15/15, seed 42, deduplicated on cleaned text
  (9,556 tweets: 6,689 train / 1,433 val / 1,434 test). `load_raw()` repairs tweet `RQMQ0L2A`, which a stray newline splits over two lines in `Train.csv`.
- Tune on `val`; evaluate once on `test` with `evaluate()` (RMSE + macro-F1, writes `reports/results.csv`).
- Save predictions with `save_predictions()` (columns: tweet_id, text, y_true, y_pred, y_score, agreement).
- Log every training run with `log_run()`.

## Run (Colab)
1. Upload this whole folder to Google Drive at `MyDrive/formative2-vaccine-sentiment/`.
2. Put `Train.csv` and `Test.csv` in `data/raw/`.
3. Open `notebooks/01_eda.ipynb`, then `02_baselines.ipynb`, and run all cells (the first cell mounts Drive).

## Run (local)
```
pip install -r requirements.txt
python -m src.preprocessing      # creates data/processed/
jupyter notebook notebooks/
```
