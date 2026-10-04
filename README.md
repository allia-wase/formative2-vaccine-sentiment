# Formative 2: Sequential Models for Vaccine Sentiment Analysis

Research-informed comparison of sequential modelling approaches for classifying vaccine-related tweets
as negative (-1), neutral (0) or positive (1).
Dataset: [To Vaccinate or Not to Vaccinate (Zindi)](https://zindi.world/competitions/to-vaccinate-or-not-to-vaccinate)

## Project goal

Compare classical, neural-from-scratch and tweet-pretrained approaches on the same fixed split, using RMSE and macro-F1. The shared pipeline keeps preprocessing, metrics and prediction files consistent so models can be compared and errors analysed together.

## Dataset and split

- Source: Zindi vaccine tweets with stance labels -1 / 0 / 1 and annotator agreement.
- Shared stratified split (seed 42, 70/15/15, deduplicated on cleaned text): **6689 train / 1433 validation / 1434 test** (9556 tweets after cleaning).
- Load only with `load_splits()` from `src/preprocessing.py`. Do not re-split.

## Approaches

1. TF-IDF + Logistic Regression / Ridge (baseline 1) — bag-of-words strength and class weighting.
2. fastText (baseline 2) — subword embeddings trained on this corpus.
3. BiLSTM + GloVe-Twitter — sequential context with pretrained Twitter word vectors.
4. Transformer encoder from scratch — self-attention without a pretrained encoder.
5. Fine-tuned BERTweet — tweet-pretrained transformer (COVID-Twitter-BERT was considered; BERTweet is the selected Model 5).

## BERTweet (Model 5)

Notebook `06_bertweet.ipynb` and helpers in `src/bert_utils.py`.

- **E1:** raw vs clean text (at lr 2e-5).
- **E2:** learning rates 1e-5 / 2e-5 / 3e-5 on raw text.
- **E3:** class weights off vs on at lr 3e-5.
- **E4:** frozen-encoder probe (comparison only).

Final configuration: raw text, lr 3e-5, class weights on, encoder fine-tuned. Test metrics come from the seed-42 run of that configuration.

## VADER side experiment

Notebook `07_vader.ipynb` scores compound sentiment on lightly cleaned `safe_text` and maps it to -1/0/1. It shows that lexicon sentiment is not the same as vaccine stance on this validation split.

## Final test results

Point estimates and 95% bootstrap intervals from `reports/final_table.csv` (one training seed per model on the shared test set):

| Model | Macro-F1 | Macro-F1 95% CI | RMSE | RMSE 95% CI |
|---|---|---|---|---|
| tfidf_logreg | 0.6551 | [0.6257, 0.6836] | 0.5838 | [0.5615, 0.6075] |
| fasttext | 0.6048 | [0.5745, 0.6360] | 0.6161 | [0.5857, 0.6480] |
| bilstm_glove | 0.6505 | [0.6188, 0.6793] | 0.5977 | [0.5659, 0.6297] |
| transformer_scratch | 0.6542 | [0.6245, 0.6817] | 0.6029 | [0.5743, 0.6312] |
| bertweet | 0.7368 | [0.7074, 0.7629] | 0.5567 | [0.5196, 0.5933] |

## Repo layout

```
data/raw/          Train.csv, Test.csv (not committed)
data/processed/    shared train/val/test split (committed)
data/embeddings/   GloVe-Twitter download + small vector caches (not committed; created by notebook 03)
src/               preprocessing.py, evaluation.py (shared)
                   nn_utils.py, models.py, experiments.py (neural models 3 and 4)
                   bert_utils.py, analysis.py (Model 5, error analysis, comparison helpers)
notebooks/         01_eda … 10_models34_followup
models/            saved model weights (not committed)
reports/           figures/, predictions/, probabilities/, results.csv, experiment_log.csv, report.md
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
- **Follow-up checks** (`notebooks/10_models34_followup.ipynb`): T6 (positional encoding re-tested on the final Transformer config), word-shuffle test, test scores over 3 training seeds, ROC/PR curves and a validation-tuned negative-class offset for the BiLSTM.
- **Report:** related work, and the methodology and results for Models 3 and 4.

### Person 3 – Keaane: Model 5, error analysis, VADER, final comparison
- **Model 5: BERTweet** (`notebooks/06_bertweet.ipynb`, `src/bert_utils.py`): experiments E1–E4, final test evaluation and predictions.
- **VADER side experiment** (`notebooks/07_vader.ipynb`): sentiment versus stance.
- **Error analysis** (`notebooks/08_error_analysis.ipynb`, `src/analysis.py`): agreement bins, heuristic slices, manual coding sample of shared hard errors.
- **Final comparison** (`notebooks/09_final_comparison.ipynb`): bootstrap intervals, paired tests, confusion and learning-curve panels.
- **Report:** Section 3.5 (BERTweet), Section 4 (error analysis), comparison discussion, VADER paragraph (`reports/report.md`).

## Shared rules (everyone)
- Load data with `load_splits()`; never re-split. Split is stratified 70/15/15, seed 42, deduplicated on cleaned text
  (9,556 tweets: 6,689 train / 1,433 val / 1,434 test). `load_raw()` repairs tweet `RQMQ0L2A`, which a stray newline splits over two lines in `Train.csv`.
- Tune on `val`; evaluate once on `test` with `evaluate()` (RMSE + macro-F1, writes `reports/results.csv`).
- Save predictions with `save_predictions()` (columns: tweet_id, text, y_true, y_pred, y_score, agreement).
- Log every training run with `log_run()`.

## Setup and run order

### Local (CPU fine for EDA, baselines, VADER, analysis)
```
pip install -r requirements.txt
# Place Train.csv and Test.csv in data/raw/ if processed splits are not already present
python -m src.preprocessing      # creates data/processed/ if needed
jupyter notebook notebooks/
```

Suggested order:
1. `01_eda.ipynb`
2. `02_baselines.ipynb`
3. `03_bilstm_glove.ipynb` (downloads GloVe-Twitter once into `data/embeddings/`, about 1.5 GB)
4. `04_transformer_scratch.ipynb`
5. `05_neural_comparison.ipynb`
6. `06_bertweet.ipynb` (GPU recommended)
7. `07_vader.ipynb`
8. `08_error_analysis.ipynb` (uses saved predictions only)
9. `09_final_comparison.ipynb` (uses saved predictions and figures only)
10. `10_models34_followup.ipynb` (Models 3 and 4: positional encoding on the final Transformer config, word-shuffle test, test scores over 3 seeds, ROC/PR curves, negative-class offset; does not change `results.csv` or the prediction files)

### Colab (T4 GPU for neural models)
1. Upload this whole folder to Google Drive at `MyDrive/formative2-vaccine-sentiment/`.
2. Put `Train.csv` and `Test.csv` in `data/raw/`.
3. Open `notebooks/01_eda.ipynb`, then `02_baselines.ipynb`, and run all cells (the first cell mounts Drive).
4. For the neural models set Runtime -> Change runtime type -> **T4 GPU**, then run `03_bilstm_glove.ipynb`,
   `04_transformer_scratch.ipynb` and `05_neural_comparison.ipynb` in that order (about 15 min each on a T4;
   notebook 03 downloads GloVe-Twitter once, about 1.5 GB). They also run on a CPU, in about 2 h each.
5. Run `06_bertweet.ipynb` on a T4 for the full three-seed sweeps. Then run `07`–`09` (CPU is enough).

### Reproducing figures
Confusion matrices and learning curves are written by `evaluate()` / `plot_history()` during model notebooks into `reports/figures/`. Error-analysis and final-comparison figures are written by notebooks 08 and 09 from the saved prediction CSVs. BERTweet sweep figures (`e1_`…`e4_`) come from notebook 06. No retraining is needed to regenerate 08/09 figures if predictions already exist.

## Demo video

Link: [Demo video (Google Drive)](https://drive.google.com/file/d/16VLErMAq6T_o4ArThgMFHdiHZ2rGLshZ/view?usp=sharing)

## Team contributions

See the [contribution tracker (Google Sheets)](https://docs.google.com/spreadsheets/d/1Ew4xySneB4duNVBXosSAAYBvDqTKz5iwbxUcJd0eHgM/edit?usp=sharing).
