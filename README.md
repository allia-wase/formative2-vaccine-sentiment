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
data/embeddings/   GloVe-Twitter download + small vector caches (not committed; created by notebook 03)
src/               preprocessing.py, evaluation.py (shared by everyone)
                   nn_utils.py, models.py, experiments.py (neural models 3 and 4)
notebooks/         01_eda, 02_baselines, 03_bilstm_glove, 04_transformer_scratch, 05_neural_comparison
models/            saved model weights (not committed)
reports/           figures/, predictions/, results.csv, experiment_log.csv
```

## Team and contributions

### Person 1 – Alliane Umutoniwase: Data & Baselines
- **EDA** (`notebooks/01_eda.ipynb`): class balance, annotator agreement, tweet length, vocabulary, noise (users, URLs, hashtags, emojis), duplicates, negation, and the tweet (`RQMQ0L2A`) that a stray newline splits over two lines.
- **Shared pipeline** (`src/`): cleaning, the fixed stratified train/val/test split, the prediction-output format and the shared experiment log.
- **Baseline 1:** TF-IDF + Logistic Regression / Ridge (experiments B1–B5).
- **Baseline 2:** fastText (experiments F1–F2) (`notebooks/02_baselines.ipynb`).
- **Report:** dataset and EDA section, data preparation, evaluation metrics, and methodology and results for both baselines.

### Person 2 – Tresor Shingiro: Literature + Models 3 & 4
- **Shared neural code** (`src/nn_utils.py`, `src/models.py`, `src/experiments.py`): tokenisation (splits `!`/`?`, maps words to GloVe-Twitter spellings such as "don't" → "dont"), GloVe download and caching, training loop with early stopping on validation macro-F1, a seeded experiment runner (3 seeds per config, every run logged), and negation / agreement slice analysis.
- **Model 3: BiLSTM + GloVe-Twitter** (`notebooks/03_bilstm_glove.ipynb`, experiments L1–L5): embeddings (random / frozen / fine-tuned), direction and pooling (LSTM vs BiLSTM; last / max / attention), class weighting, capacity and dropout, GloVe dimension.
- **Model 4: Transformer encoder from scratch** (`notebooks/04_transformer_scratch.ipynb`, experiments T1–T5): positional encoding (none / sinusoidal / learned, which tests whether word order matters), model size, word vs BPE subword tokens, GloVe initialisation, pooling and warm-up.
- **Comparison** (`notebooks/05_neural_comparison.ipynb`): test results, slices and seed stability for all models.
- **Report:** related work, and the methodology and results for Models 3 and 4.

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
4. For the neural models set Runtime -> Change runtime type -> **T4 GPU**, then run `03_bilstm_glove.ipynb`,
   `04_transformer_scratch.ipynb` and `05_neural_comparison.ipynb` in that order (about 15 min each on a T4;
   notebook 03 downloads GloVe-Twitter once, about 1.5 GB). They also run on a CPU, in about 2 h each.

## Run (local)
```
pip install -r requirements.txt
python -m src.preprocessing      # creates data/processed/
jupyter notebook notebooks/
```
