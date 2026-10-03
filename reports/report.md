# Formative 2 report: vaccine-tweet stance classification

This file holds the sections owned by Keaane (Model 5, error analysis, comparison, VADER). Sections owned by other teammates (introduction, related work, data and EDA, Models 1 to 4 methodology and results) are not written here and should be merged from their drafts when available.

Numbers below were recomputed from the cited sources. Seed-42 test predictions are a single training run per model unless noted.

## 3.5 Model 5: BERTweet


### Method

Model 5 fine-tunes `vinai/bertweet-base` for three-way stance classification on the shared tweet split (train 6689, validation 1433, test 1434). Inputs use the raw `safe_text` field with `<user>` / `<url>` mapping for the tokenizer. Sequences are truncated or padded to length 80. A linear classification head produces logits over labels {-1, 0, 1}. Training uses AdamW with linear warmup (10% of steps) then decay, mixed precision on CUDA, gradient clipping at 1.0, batch size 16, up to 5 epochs, and early stopping on validation macro-F1 with patience 2. Each configuration is run with three seeds (42, 1, 2). Selection uses mean validation macro-F1; the seed-42 checkpoint of the chosen configuration is evaluated once on the test set. Method details are taken from `src/bert_utils.py` (`MAX_LEN = 80`, AdamW, warmup, amp, clip, patience) and notebook 06.

### Experiments (validation macro-F1, mean of three seeds)

Means below were recomputed from `reports/experiment_log.csv` for `model == "bertweet"`.

| Experiment | Setting | Seeds used | Mean val macro-F1 | Per-seed values |
|---|---|---|---|---|
| Text (E1, lr 2e-5, no class weights) | raw | 3 | 0.7274 | [0.7298, 0.7261, 0.7263] |
| Text (E1, lr 2e-5, no class weights) | clean | 3 | 0.7273 | [0.725, 0.7308, 0.7261] |
| Learning rate (raw; 2e-5 from E1, others from E2) | 1e-5 | 3 | 0.7105 | [0.7103, 0.712, 0.7092] |
| Learning rate | 2e-5 | 3 | 0.7274 | [0.7298, 0.7261, 0.7263] |
| Learning rate | 3e-5 | 3 | 0.7297 | [0.7145, 0.7387, 0.7359] |
| Class weights (E3 vs E2 at lr 3e-5) | off (E2) | 3 | 0.7297 | [0.7145, 0.7387, 0.7359] |
| Class weights | on (E3) | 3 | 0.7389 | [0.7432, 0.7359, 0.7376] |
| Frozen-encoder probe (E4; comparison only) | freeze, lr 0.001 | 3 | 0.5872 | [0.587, 0.5815, 0.5932] |

Raw and clean text are effectively tied at lr 2e-5 (0.7274 vs 0.7273); that text comparison was run only at lr 2e-5, not at the final 3e-5. Among learning rates on raw text, 3e-5 has the highest mean (0.7297). Turning on class weights at lr 3e-5 raises the mean from 0.7297 to 0.7389, a gain of about 0.009, which is comparable to the E3 seed std of 0.0038, so it is a small gain rather than a clear one. The frozen-encoder probe is much lower (0.5872) and is reported only as a comparison.

### Final configuration and test results

Final configuration: raw text, learning rate 3e-5, class weights on, encoder not frozen (E3). The seed-42 validation macro-F1 for that run is 0.7432 (`reports/experiment_log.csv`).

Test metrics for BERTweet from `reports/final_table.csv` (point estimates and 95% bootstrap intervals):

| Metric | Point | 95% CI |
|---|---|---|
| Macro-F1 | 0.7368 | [0.7074, 0.7629] |
| RMSE | 0.5567 | [0.5196, 0.5933] |
| Accuracy | 0.7908 | [0.7685, 0.8110] |
| F1 negative | 0.5811 | [0.5123, 0.6436] |
| F1 neutral | 0.8243 | [0.8009, 0.8457] |
| F1 positive | 0.8051 | [0.7786, 0.8293] |

These test numbers come from one seed-42 training run.

## 4. Error analysis

Using the five aligned test prediction files (`reports/predictions/*.csv`, n = 1434):

- All five models correct: 791 tweets.
- All five models wrong: 128 tweets.
- Mean number of models wrong per tweet: 1.3264.

### Annotator agreement

| Agreement | n | BERTweet macro-F1 | BiLSTM macro-F1 | TF-IDF LogReg macro-F1 |
|---|---|---|---|---|
| 1.0 | 851 | 0.8649 | 0.7554 | 0.7045 |
| 0.667 | 549 | 0.6096 | 0.5416 | 0.5622 |
| 0.333 | 34 | 0.1138 | 0.0702 | 0.1739 |

Performance falls as agreement falls. The 0.333 bin is only 34 tweets (2.4% of the test set), so those scores are noisy.

### Heuristic slices

Slice sizes on the test set: negation 366, sarcasm cues 182, news-style 406. These are keyword or pattern heuristics, not gold labels. The sarcasm-cue pattern is a broadened cue list; it is large enough to score (n = 182) but remains heuristic.

Macro-F1 on the negation slice is below each model's overall score. Drops (overall minus negation): TF-IDF LogReg 0.0423, fastText 0.0583, BiLSTM 0.0460, transformer 0.0405, BERTweet 0.0121. News-style scores stay close to overall (for example BERTweet 0.7252 vs 0.7368 overall).

### BERTweet confusion (true negative class)

From the test predictions, among 152 true-negative tweets the row-normalised rates are: negative to neutral 0.1447, negative to positive 0.2895. The negative-to-positive rate is lower than for the other models (0.3553 to 0.5197). The negative-to-neutral rate is not lower than TF-IDF LogReg (0.0724) or the scratch transformer (0.0789); it is lower than fastText (0.2303) and similar to the BiLSTM (0.1513).

### Manual coding of 30 shared hard errors

Thirty tweets that all five models miss were sampled with seed 42 and coded by hand into five categories (illustrative counts, not population rates):

| Category | Count |
|---|---|
| other | 10 |
| sarcasm | 6 |
| news_style | 6 |
| ambiguous_label | 6 |
| negation | 2 |

Within this sample, using BERTweet as `y_pred`, the (y_true, y_pred) pair counts are: sarcasm {(-1,1): 3, (0,1): 1, (1,0): 2}; other {(-1,0): 2, (0,1): 5, (0,-1): 2, (1,-1): 1}; news_style {(-1,0): 1, (-1,1): 1, (0,1): 2, (1,0): 2}; ambiguous_label {(0,1): 5, (1,-1): 1}; negation {(0,-1): 1, (1,0): 1}.

Illustrative examples (shortened; no offensive wording):

1. ambiguous_label (TE40NOVB, y_true = 0): argues for vaccinating only those at risk rather than everyone; the neutral gold label is debatable.
2. news_style (RE4YLV4S, y_true = -1): AM-News headline about health conspiracy theories with a URL; news form with low agreement.
3. sarcasm (XLX3D9RE, y_true = -1): rhetorical challenge ("What's wrong with getting measles?") defending exemption.
4. negation (GIGS057L, y_true = 0): "there was no such thing as a chicken pox vaccine when we were kids".

### Limitations

Test predictions for each model are a single training seed. Several slices are small or pattern-based (especially agreement 0.333 with n = 34, and keyword sarcasm/negation detection). Only 30 all-wrong tweets were coded by hand; those counts are illustrative, not estimated rates. Label noise is visible at low agreement.

## Comparison discussion

Point macro-F1 on the shared test set (`reports/final_table.csv`): BERTweet 0.7368, TF-IDF LogReg 0.6551, scratch transformer 0.6542, BiLSTM 0.6505, fastText 0.6048. All of these test numbers are from one seed per model. The four models with no pretrained transformer encoder span 0.6048 to 0.6551, about 0.05 macro-F1; the BiLSTM still uses GloVe-Twitter embeddings. Among TF-IDF LogReg, BiLSTM and the scratch transformer the gaps are smaller (within about 0.005).

Paired bootstrap of macro-F1(BERTweet) − macro-F1(other) on the test tweets (notebook 09, 2000 resamples, seed 42) gives mean differences 0.0817 vs LogReg, 0.1316 vs fastText, 0.0864 vs BiLSTM, and 0.0820 vs the scratch transformer; all four 95% intervals lie above zero (lower ends 0.0507, 0.0977, 0.0548, 0.0498) and share_le_0 = 0 in each case. Training variation is not included in that bootstrap (E3 validation macro-F1 std across three seeds is 0.0038).

The largest BERTweet gain relative to the others on the class-wise F1 table is on the negative class: 0.5811 against 0.3262–0.4638 for the other four models (`reports/final_table.csv`).

## VADER: sentiment is not stance

On the validation split, VADER compound mapped with the standard ±0.05 thresholds (version A) reaches macro-F1 0.3242; a train-tuned threshold of 0.30 (version B) reaches 0.3444 (`reports/vader_val_results.csv`). RMSE of compound against the stance label is 0.8573; Spearman correlation is −0.0058 (p = 0.8254). Always predicting neutral scores 0.2171 macro-F1; a random predictor matched to validation class proportions averages 0.3330 over 500 draws with seed 42 (notebook 07). Version A is near that random reference; version B is only slightly above it. A lexicon sentiment score does not recover vaccine stance on this split.



### Figures (BERTweet, error analysis, comparison, VADER)

![BERTweet text ablation](figures/e1_bertweet_text.png)

![BERTweet learning-rate sweep](figures/e2_bertweet_lr.png)

![BERTweet class-weight sweep](figures/e3_bertweet_class_weight.png)

![BERTweet freeze probe](figures/e4_bertweet_freeze.png)

![BERTweet confusion matrix](figures/cm_bertweet.png)

![BERTweet learning curves](figures/curves_bertweet.png)

![Test macro-F1 with bootstrap intervals](figures/final_macro_f1.png)

![Confusion matrices for all five models](figures/final_confusion_grid.png)

![Learning-curve panel](figures/final_learning_curves.png)

![Macro-F1 by annotator agreement](figures/error_agreement.png)

![Macro-F1 on heuristic slices](figures/error_slices.png)

![VADER validation confusion](figures/vader_confusion.png)

![VADER compound by true stance](figures/vader_compound_by_label.png)

## Conclusion

On the shared test set of 1434 tweets, BERTweet reaches macro-F1 0.7368 and RMSE 0.5567, ahead of TF-IDF LogReg (0.6551 / 0.5838), the scratch transformer (0.6542 / 0.6029), the BiLSTM (0.6505 / 0.5977) and fastText (0.6048 / 0.6161). Paired bootstrap differences of BERTweet against each other model stay positive, with lower interval ends about 0.05 or higher, though each test score is from one training seed. The clearest class-wise gain is on negatives (F1 0.5811 versus 0.3262 to 0.4638). Error analysis shows 791 tweets correct for all five models and 128 wrong for all five; performance falls with annotator agreement, and heuristic negation tweets cost every model some macro-F1. A thirty-tweet hand-coded sample of shared hard errors is illustrative only. VADER validation macro-F1 stays near a random-by-proportions reference (0.3242 to 0.3444 versus 0.3330), so lexicon sentiment does not stand in for stance here. Taken together, a tweet-pretrained encoder helps most on this split, while models with no pretrained transformer encoder remain within about 0.05 macro-F1 of each other (the BiLSTM still uses GloVe-Twitter embeddings).

## Numbers used


| Number | Value | Source |
|---|---|---|
| Train / val / test sizes | 6689 / 1433 / 1434 | `data/processed/*.csv` via `load_splits()` |
| BERTweet max length | 80 | `src/bert_utils.py` (`MAX_LEN`) |
| Warmup / clip / patience / max epochs / batch | 0.1 / 1.0 / 2 / 5 / 16 | `src/bert_utils.py`; notebook 06 (`BATCH = 16`) |
| E1 raw mean val macro-F1 | 0.7274 ([0.7298, 0.7261, 0.7263]) | `reports/experiment_log.csv` bertweet E1 raw |
| E1 clean mean val macro-F1 | 0.7273 ([0.725, 0.7308, 0.7261]) | `reports/experiment_log.csv` bertweet E1 clean |
| lr 1e-5 mean val macro-F1 | 0.7105 ([0.7103, 0.712, 0.7092]) | `reports/experiment_log.csv` bertweet E2 |
| lr 2e-5 mean val macro-F1 | 0.7274 ([0.7298, 0.7261, 0.7263]) | `reports/experiment_log.csv` bertweet E1 raw |
| lr 3e-5 mean val macro-F1 | 0.7297 ([0.7145, 0.7387, 0.7359]) | `reports/experiment_log.csv` bertweet E2 |
| E3 class-weight-on mean val macro-F1 | 0.7389 ([0.7432, 0.7359, 0.7376]) | `reports/experiment_log.csv` bertweet E3 |
| E4 freeze mean val macro-F1 | 0.5872 ([0.587, 0.5815, 0.5932]) | `reports/experiment_log.csv` bertweet E4 |
| E3 seed-42 val macro-F1 / RMSE | 0.7432 / 0.5616 | `reports/experiment_log.csv` |
| E3 val macro-F1 std | 0.0038 | notebook 09 paired cell; also std of E3 log rows |
| BERTweet test macro-F1 / CI | 0.7368 / [0.7074, 0.7629] | `reports/final_table.csv` |
| BERTweet test RMSE / CI | 0.5567 / [0.5196, 0.5933] | `reports/final_table.csv` |
| BERTweet test accuracy / CI | 0.7908 / [0.7685, 0.8110] | `reports/final_table.csv` |
| BERTweet F1 −/0/+ | 0.5811 / 0.8243 / 0.8051 | `reports/final_table.csv` |
| Other models test macro-F1 | 0.6551, 0.6048, 0.6505, 0.6542 | `reports/final_table.csv` |
| Other models F1 negative | 0.4638, 0.3262, 0.4130, 0.4514 | `reports/final_table.csv` |
| All-correct / all-wrong / mean wrong | 791 / 128 / 1.3264 | recomputed from `reports/predictions/*.csv` (matches notebook 08) |
| Agreement bin sizes | 851 / 549 / 34 | notebook 08 / recomputed |
| Agreement 1.0 macro-F1 (BT / BiLSTM / LogReg) | 0.8649 / 0.7554 / 0.7045 | notebook 08 / recomputed |
| Agreement 0.667 macro-F1 (BT / BiLSTM / LogReg) | 0.6096 / 0.5416 / 0.5622 | notebook 08 / recomputed |
| Agreement 0.333 macro-F1 (BT / BiLSTM / LogReg) | 0.1138 / 0.0702 / 0.1739 | notebook 08 / recomputed |
| Slice sizes neg / sarc / news | 366 / 182 / 406 | notebook 08 / recomputed |
| Negation macro-F1 drops | 0.0423, 0.0583, 0.0460, 0.0405, 0.0121 | recomputed from predictions |
| News-style macro-F1 (LogReg / fastText / BiLSTM / transformer / BERTweet) | 0.6644 / 0.5870 / 0.6520 / 0.6453 / 0.7252 | recomputed from predictions |
| Class-weight gain (E3 mean minus E2 lr 3e-5 mean) | 0.0092 (about 0.009); E3 seed std 0.0038 | `reports/experiment_log.csv` |
| BERTweet neg→neu / neg→pos | 0.1447 / 0.2895 | recomputed; notebook 09 |
| Other neg→pos range | 0.3553–0.5197 | recomputed |
| LogReg / transformer / fastText / BiLSTM neg→neu | 0.0724 / 0.0789 / 0.2303 / 0.1513 | recomputed |
| Manual coding counts | other 10, sarcasm 6, news_style 6, ambiguous_label 6, negation 2 | `reports/error_sample_for_manual_coding.csv` |
| Paired BT diffs / lo | 0.0817/0.0507, 0.1316/0.0977, 0.0864/0.0548, 0.0820/0.0498 | notebook 09 outputs |
| VADER A/B val macro-F1 | 0.3242 / 0.3444 | `reports/vader_val_results.csv` |
| VADER val RMSE / Spearman | 0.8573 / −0.0058 | `reports/vader_val_results.csv` |
| Always-neutral / random mean F1 | 0.2171 / 0.3330 | notebook 07 outputs |

