# Complete Presentation Guide, Theoretical Mastery & Speaker Notes (Slides 1–8)
**Project:** PAN 2025 Shared Task 1 — Generative AI Detection & Authorship Attribution  
**Author:** Elia Gismondi  
**Reference LaTeX Slide Deck:** [`reports/presentation_slides.tex`](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/presentation_slides.tex)

---

## Table of Contents
1. [Slide 1 — Title, Scope & Dual-Task Architecture](#slide-1--title-scope--dual-task-architecture)
2. [Slide 2 — Steps 1 & 2: Multi-Level Feature Space ($D = 4,600$) & Corpus Execution](#slide-2--steps-1--2-multi-level-feature-space-d--4600--corpus-execution)
3. [Slide 3 — Step 3A: Feature Influence on Binary Output ($y \in \{0, 1\}$)](#slide-3--step-3a-feature-influence-on-binary-output-y-in-0-1)
4. [Slide 4 — Step 3B: Discriminating Across 22 AI Models (Multiclass ANOVA)](#slide-4--step-3b-discriminating-across-22-ai-models-multiclass-anova)
5. [Slide 5 — Step 3C: Feature Family Ablation & Leave-One-Out Impact](#slide-5--step-3c-feature-family-ablation--leave-one-out-impact)
6. [Slide 6 — Step 4A: Dual-Task Baseline Models Benchmark ($N_{\text{val}} = 3,589$)](#slide-6--step-4a-dual-task-baseline-models-benchmark-n_textval--3589)
7. [Slide 7 — Step 4B: Confusion Matrices & Cross-Domain Generalization](#slide-7--step-4b-confusion-matrices--cross-domain-generalization)
8. [Slide 8 — Step 4C: Per-Author & Per-LLM Detection and Attribution](#slide-8--step-4c-per-author--per-llm-detection-and-attribution)
9. [Quick Mathematical & Conceptual Glossary for Defense Q&A](#9-quick-mathematical--conceptual-glossary-for-defense-qa)

---

## Slide 1 — Title, Scope & Dual-Task Architecture

### 1. Deep Context & Subject Mastery
This presentation covers the four methodological steps requested after our initial Exploratory Data Analysis (EDA):
1. **Extract features**: Transform unstructured raw texts into an interpretable numerical vector space.
2. **Run them**: Execute the feature pipeline across all **27,296 documents** in the PAN 2025 corpus (`23,707` Training + `3,589` Validation texts) across three literary domains (*Fiction*, *Essays*, *News*).
3. **Measure their influence for output**: Mathematically quantify how much each individual feature and feature family drives the model's predictions.
4. **Run the baseline model**: Train, cross-validate, and benchmark canonical baseline classifiers from the authorship attribution literature.

**Key Value Proposition (Dual-Task Formulation):**
Rather than stopping at binary classification (*Human `0` vs. AI `1`*, **Task 1: Binary Detection**), we leveraged the corpus metadata to solve the much harder problem of **LLM Fingerprinting / Model Attribution (Task 2)**: predicting **which specific author or LLM among 23 candidates** (`1` human class + `22` distinct LLMs spanning 6 major vendor families: OpenAI, Google Gemini/PaLM, Meta LLaMA, Mistral AI, DeepSeek/Qwen, and TII Falcon) generated each text.

### 2. Speaker Note (Ready to Deliver)
> *"Good morning everyone. In this presentation, we cover the four core steps of our experimental pipeline for the PAN 2025 Shared Task: feature extraction, full-corpus vectorization, statistical feature influence analysis, and baseline model evaluation. Importantly, we framed our pipeline as a Dual-Task system: first, Binary Detection to separate Human from AI-generated texts across 27,296 documents; and second, fine-grained Model Attribution across 23 candidate classes, identifying which specific LLM architecture wrote each text."*

### 3. Potential Professor Questions & Answers (Q&A)
- **Q: Why did you frame the problem as a Dual-Task system instead of binary only?**
  - *A:* Because in forensic stylometry, detecting that a text is synthetic is only the first step; attributing the text to a specific generator family (OpenAI vs. LLaMA vs. Gemini) tests whether our stylometric and N-gram features capture deep architectural decoding signatures rather than superficial binary artifacts.

---

## Slide 2 — Steps 1 & 2: Multi-Level Feature Space ($D = 4,600$) & Corpus Execution

### 1. Deep Context & Subject Mastery
- **Why a 4-Family Multi-Level Feature Space?**
  In authorship attribution literature (Stamatatos, 2009), relying solely on content words causes **topic confounding** (the classifier learns *what* the text is about rather than *who* wrote it). To guarantee topic-invariance, our **4,600-dimensional vector space** combines four complementary levels:
  1. **Stylometry & Lexical Richness (`30` dense features):** Captures macro-structural rhythm and vocabulary diversity. It includes classical metrics (`ttr`, word/sentence lengths, 9 specific punctuation densities) and length-invariant mathematical indices such as **Yule's $K$**, **Brunet's $W$** ($W = N^{V^{-0.172}}$), and **Honoré's $R$** ($R = \frac{100 \ln N}{1 - V_1/V}$, which measures the richness of *hapax legomena*—words occurring only once). It also includes custom syntactic ratios (`functional_lexical_ratio`, `expressive_punct_ratio`, `syntactic_pause_ratio`).
  2. **Function Words (`70` dense features):** Relative frequencies of 70 closed-class grammatical words (prepositions, articles, auxiliaries, conjunctions: *the, of, and, to, so, was...*). These are **topic-invariant**: whether an author writes about sports, politics, or fiction, their unconscious grammatical scaffolding remains constant.
  3. **Character Sub-Word N-Grams (`3,000` sparse TF-IDF features):** Boundary-aware character 3-to-4-grams (`analyzer='char_wb'`). They capture morphological prefixes, suffixes (`-ing`, `-tion`, `-ly`), word stems, and punctuation habits. We apply `sublinear_tf=True` (replacing raw term frequency $tf$ with $1 + \ln(tf)$) so that repeated words in long texts do not distort the vector norm.
  4. **Word N-Grams (`1,500` sparse TF-IDF features):** Lexical unigrams and bigrams preserving stop-words to capture short syntactic collocations (*"said the"*, *"as well"*).
- **Strict Data Leakage Prevention:**
  Both the `StandardScaler` (applied to the 100 dense features) and the `TfidfVectorizer` vocabularies are fitted (`fit()`) **strictly on the 23,707 Training texts** and subsequently applied (`transform()`) to the 3,589 Validation texts.
- **How to Read the Visual (`stylometric_distributions`):**
  Shows Kernel Density Estimation (KDE) curves for the 3 core stylometric anchors:
  - *Left (TTR):* The blue curve (AI) is shifted right relative to the green curve (Human) $\to$ LLMs exhibit higher local lexical diversity (`0.522` vs. `0.471`) because decoding repetition penalties discourage reusing the same token within a context window.
  - *Center (Punctuation Density):* The green curve (Human) is shifted right $\to$ human authors use ~32% more punctuation (`0.0375` vs. `0.0284`, driven by dialogue, dashes, and semicolons).
  - *Right (Average Word Length):* The blue curve (AI) is strongly shifted toward longer words (`5.02` vs. `4.34` chars) $\to$ reflecting formal, academic, Latinate vocabulary preferences in LLMs.

### 2. Speaker Note (Ready to Deliver)
> *"In Steps 1 and 2, we engineered a 4,600-dimensional feature space combining four topic-invariant families: 30 dense stylometric and lexical richness indices, 70 closed-class function words, 3,000 sub-word character 3-to-4-grams, and 1,500 word 1-to-2-grams with sublinear TF-IDF scaling. To strictly prevent data leakage, all scalers and vocabularies were fitted exclusively on the 23,707 training texts and applied to the 3,589 validation texts. The extraction runs at over 415 documents per second with zero NaN or infinite values. As shown in the KDE plots on the right, AI texts exhibit systematically higher lexical diversity (TTR) and longer average word lengths, whereas human authors use significantly denser punctuation."*

### 3. Potential Professor Questions & Answers (Q&A)
- **Q: Why did you apply `StandardScaler` only to the 100 dense features and not to the 4,500 TF-IDF N-grams?**
  - *A:* The 100 dense features have heterogeneous physical units (e.g., `word_count` reaches 4,000 while `comma_density` is around `0.01`), so z-score standardization is essential for linear models. Conversely, TF-IDF vectors are already $L_2$-normalized, and mean-centering them would destroy the 63.2% sparsity of the CSR matrix, increasing memory usage dramatically.

---

## Slide 3 — Step 3A: Feature Influence on Binary Output ($y \in \{0, 1\}$)

### 1. Deep Context & Subject Mastery
- **How We Measured Feature Influence on the Binary Output:**
  We evaluated feature influence at two complementary levels across all 23,707 training texts:
  1. **Univariate Statistical Influence (Left Panel — Pearson $r$, Spearman $\rho$, ANOVA $F$, Mutual Information):**
     Evaluates each dense feature independently against $y$ (`0` = Human, `1` = AI):
     - The single strongest predictor is **`functional_lexical_ratio` ($r = -0.613$, ANOVA $F = 14,234$)**: the ratio of closed-class function words to open-class content words. It is negative (green = Human) because spontaneous human prose relies on a much denser grammatical scaffolding, whereas LLMs pack high-density lexical content.
     - Second is **`semicolon_density` ($r = -0.582$, Mutual Info $= 0.277$)**: standard LLMs almost never generate semicolons unless explicitly prompted, whereas human writers use them regularly to link clauses.
     - On the AI side (blue, $r > 0$), **`avg_word_len` ($r = +0.576$)** and **`word_len_std` ($r = +0.562$)** dominate.
  2. **Multivariate Model-Based Influence (Right Panel — Regularized L2 Logistic Regression Weights $\beta_j$ across all 4,600 dims + Permutation Importance):**
     Evaluates all 4,600 features jointly inside a predictive classifier:
     - **Permutation Importance:** We measured how much Validation Macro F1 drops when randomly shuffling a single feature column. Shuffling `avg_word_len` causes a massive **$-14.1\%$ drop in Macro F1**!
     - **RLHF Lexical Hallmarks:** The highest positive weights $\beta_j > 0$ belong to quintessential RLHF/instruction-tuned words: ***"significant"* ($\beta = +2.03$), *"ongoing"* ($\beta = +1.54$), *"impact"* ($\beta = +1.53$), *"despite"* ($\beta = +1.46$), and *"crucial"* ($\beta = +1.42$)**. Conversely, human causal and narrative connectors such as ***"because"* ($\beta = -2.09$), *"although"* ($\beta = -1.83$), and *"said the"* ($\beta = -1.49$)** strongly push predictions toward Human authorship.

- **Advanced Technical Detail (Why `functional_lexical_ratio` is negative on the left but positive on the right):**
  Univariately (left), `functional_lexical_ratio` correlates negatively with AI ($r = -0.613$). However, inside the multivariate model (right), all **70 individual function words** (`the`, `and`, `of`, `is`, `was`) are simultaneously present with large negative weights ($\beta \in [-1.47, -0.88]$). Because `functional_lexical_ratio` is the aggregate sum of those exact same 70 function words and is strongly collinear with `avg_word_len` ($r = -0.82$), the L2 linear model uses it as a **collinear suppressor/balancing variable** to avoid double-counting the 70 individual function word weights!

### 2. Speaker Note (Ready to Deliver)
> *"In Step 3A, we quantified feature influence on the binary target using both univariate statistics on the left and multivariate regularized Logistic Regression weights on the right. Univariately, the strongest human markers are functional-to-lexical ratio ($r = -0.613$) and semicolon density ($r = -0.582$), while average word length ($r = +0.576$) is the primary AI marker—causing a 14.1% drop in Macro F1 under permutation importance. On the right, inspecting the global 4,600-dimensional model reveals classic RLHF vocabulary biases: words like 'significant', 'ongoing', 'impact', 'despite', and 'crucial' carry the highest positive weights toward AI, whereas causal connectors like 'because' and 'although' strongly predict human authorship."*

---

## Slide 4 — Step 3B: Discriminating Across 22 AI Models (Multiclass ANOVA)

### 1. Deep Context & Subject Mastery
- **How Do We Measure Feature Influence Across 22 Unordered AI Classes?**
  Pearson $r$ only applies to binary or ordinal targets. For 22 distinct AI models, the rigorous statistical measure is **Multiclass One-Way ANOVA ($F$-statistic)**:
  $$F = \frac{\text{MS}_{\text{between-models}}}{\text{MS}_{\text{within-model}}}$$
  A high $F$-score means the feature takes tightly clustered values within each LLM but varies dramatically from one LLM architecture to another.
- **How to Read the Heatmap (`llm_family_stylometric_fingerprints.png`):**
  Columns represent the 7 architectural families; rows represent the top 15 stylometric features ranked by Multiclass ANOVA $F$. Cell values show the **mean standardized Z-score** relative to the corpus mean (red = above average, blue = below average):
  1. **`uppercase_ratio` ($F = 700$)**: The #1 discriminator across AI models! Mistral (`+0.33z`) and Google Gemini (`+0.21z`) use significantly more capital letters (structured headers, bulleted lists, acronyms) than TII Falcon (`-0.49z`) and OpenAI (`-0.35z`).
  2. **OpenAI Family (`gpt-4o`, `o3-mini`, `gpt-3.5-turbo`):** Exhibits the highest lexical diversity across the entire corpus (`ttr = +0.65z`, `hapax_ratio = +0.58z`, `brunet_w = -0.63z`) combined with the lowest comma density (`-0.50z`), favoring clean, direct clause structures.
  3. **DeepSeek / Qwen Family (`deepseek-r1-distill-qwen-32b`):** Because DeepSeek-R1 is distilled from long chain-of-thought reasoning traces, it generates substantially longer multi-sentence texts (`sentence_count = +0.80z`, `char_count = +0.61z`, `word_count = +0.60z`) with lower TTR (`-0.46z`).
  4. **TII Falcon (`falcon3-10b-instruct`):** Shows the highest proportion of once-occurring words (`hapax_ratio = +0.73z` and `honore_r = +0.65z`).

### 2. Speaker Note (Ready to Deliver)
> *"Moving to Step 3B, we asked: which features allow us to distinguish between the 22 different AI models? Using multiclass one-way ANOVA across the AI corpus, we ranked features by their between-model to within-model variance ratio. Uppercase ratio ($F = 700$) and lexical richness indices—Root TTR, Hapax ratio, and Brunet's W—emerge as the strongest LLM fingerprints. The standardized Z-score heatmap on the right clearly illustrates each vendor's signature: OpenAI models maximize vocabulary diversity (TTR $+0.65z$) while minimizing comma density ($-0.50z$); DeepSeek reasoning models produce significantly longer, multi-sentence outputs ($+0.80z$); and Falcon exhibits the highest hapax legomena ratio ($+0.73z$)."*

---

## Slide 5 — Step 3C: Feature Family Ablation & Leave-One-Out Impact

### 1. Deep Context & Subject Mastery
- **Why Perform an Ablation Study?**
  To prove that our 4 feature families are not redundant, we ran two systematic experiments on the Validation Set ($N = 3,589$):
  1. **Left Panel (Individual vs. Combined Feature Spaces):**
     - **Stylometry Alone (`30` dims):** Achieves **`93.61%` Binary Macro F1 (`97.91%` ROC-AUC)** using only 30 compact mathematical features! This proves that macro-structural stylometry alone captures the core Human-vs-AI separation without looking at any specific words.
     - **However, on 23-Class Model Attribution (red bars):** 30 stylometric features alone achieve `42.0%` Macro F1, whereas combining Stylometry + Function Words + Char/Word N-Grams boosts 23-class Macro F1 to **`73.27%`** (`84.12%` Top-1 Accuracy).
  2. **Right Panel (Leave-One-Family-Out — LOFO):**
     We start from the full 4,600-dimensional system and remove one family at a time:
     - Removing **Stylometry (`LOFO: w/o Stylometry`)** causes the single largest drop in 23-class LLM attribution: **$-4.76$ percentage points** in Macro F1 (orange bar on the left)!
     - Removing **Char N-Grams (`LOFO: w/o Char N-Grams`)** drops 23-class Macro F1 by **$-3.06$ percentage points**.
     - Removing **Word N-Grams** drops 23-class Macro F1 by **$-1.58$ percentage points**.
     - **Conclusion:** Global stylometric structure and local sub-word/lexical N-grams provide orthogonal, complementary information essential for fine-grained LLM attribution.

### 2. Speaker Note (Ready to Deliver)
> *"In Step 3C, we conducted a Feature Family Ablation study to measure the unique contribution of each feature group on both tasks. On the left, remarkably, using only our 30 dense stylometric features achieves 93.61% Macro F1 on binary detection. However, for 23-class model attribution—shown in red—combining dense stylometrics with character and word N-grams is essential, boosting Macro F1 from 42.0% up to 73.27%. On the right, our Leave-One-Family-Out analysis proves this synergy: dropping the 30 stylometric features from the full pipeline causes the sharpest drop in 23-class attribution (-4.76 percentage points), followed by character N-grams (-3.06 points)."*

---

## Slide 6 — Step 4A: Dual-Task Baseline Models Benchmark ($N_{\text{val}} = 3,589$)

### 1. Deep Context & Subject Mastery
- **Why We Selected These Specific Baseline Classifiers:**
  1. **Dummy Classifier (Majority Class):** Always predicts the most frequent class (`AI` in binary, `human` in 23-class). Establishes the trivial lower bound (`39.2%` Binary Macro F1; `35.6%` 23-class Top-1 Accuracy, `2.3%` 23-class Macro F1).
  2. **Complement Naive Bayes (`ComplementNB`):** Designed to correct standard Multinomial NB biases on imbalanced text datasets (fitted on non-negative `MinMaxScaler [0,1]` dense features + TF-IDF). Trains in `0.04s` and achieves a solid **`90.15%` Binary Macro F1** (`96.50%` ROC-AUC), but drops to `49.87%` Top-1 Accuracy on 23 classes because its conditional independence assumption is violated by overlapping char/word N-grams.
  3. **Logistic Regression (L1 Lasso & L2 Ridge):** Discriminative linear classifiers optimizing regularized cross-entropy loss. They reach **`97.84%` (L1) and `97.99%` (L2) Binary Macro F1** (`99.71%` ROC-AUC) and **`81.64%` Top-1 Accuracy (`94.59%` Top-3 Accuracy)** on 23-class attribution.
  4. **Linear SVM (`LinearSVC` — Stamatatos Authorship Attribution Standard):** Maximizes the margin between separating hyperplanes in high-dimensional sparse spaces. It is the top-performing model across the board:
     - **Task 1 (Binary):** **`98.88%` 5-Fold CV Macro F1** on Train and **`98.60%` Validation Macro F1 (`98.72%` Accuracy, `99.84%` ROC-AUC)**, committing only **26 False Positives and 20 False Negatives** out of 3,589 validation texts!
     - **Task 2 (23-Class Attribution):** **`84.12%` Top-1 Accuracy**, **`94.93%` Top-3 Accuracy**, **`73.27%` Macro F1**, and **`89.05%` 7-Family Accuracy**.
  5. **Two-Stage Hierarchical SVM (`Stage 1 Binary SVM -> Stage 2 22-LLM Specialist SVM`):**
     First filters Human vs. AI using the Stage 1 Binary SVM, then routes AI texts to a specialized 22-class SVM trained exclusively on AI texts. Because Stage 2 dedicates its entire capacity to intra-LLM boundaries, it achieves the **highest exact attribution accuracy on AI texts (`76.08%`)** and the **highest 7-Family Macro F1 (`83.14%`)**.

### 2. Speaker Note (Ready to Deliver)
> *"In Step 4A, we benchmarked our baseline classifiers on both tasks using 5-fold cross-validation on Train and evaluation on the 3,589 validation texts. For Binary Detection, while Complement Naive Bayes reaches 90.2% Macro F1, discriminative linear models—L1/L2 Logistic Regression and Linear SVM—exceed 98% Macro F1. Specifically, Linear SVM achieves the best binary performance with 98.60% Validation Macro F1 and 99.84% ROC-AUC, making only 46 total errors out of 3,589 texts. For Task 2—predicting which of the 23 authors or AI models wrote the text—Linear SVM and our Two-Stage Hierarchical SVM achieve 84.1% Top-1 accuracy, nearly 95% Top-3 accuracy, and 89.1% accuracy at identifying the LLM vendor family."*

---

## Slide 7 — Step 4B: Confusion Matrices & Cross-Domain Generalization

### 1. Deep Context & Subject Mastery
This slide provides three critical diagnostic views of our best baseline (`Linear SVM` / `Two-Stage SVM`):
1. **Left Panel — 2×2 Binary Confusion Matrix:**
   - Out of `1,277` true Human validation texts, `1,251` are correctly classified (**`98.0%` Specificity / True Negative Recall**) and only `26` (`2.0%`) are false positives.
   - Out of `2,312` true AI validation texts, `2,292` are correctly detected (**`99.1%` Sensitivity / True Positive Recall**) and only `20` (`0.9%`) evade detection.
2. **Center Panel — 7×7 Row-Normalized Architectural Family Confusion Matrix (%):**
   - Shows a dominant main diagonal: **Human (`98.0%`)**, **OpenAI (`92.7%`)**, **Google Gemini (`81.3%`)**, **DeepSeek/Qwen (`81.1%`)**, **TII Falcon (`78.8%`)**, **Meta LLaMA (`77.4%`)**, and **Mistral AI (`71.5%`)**.
   - Off-diagonal confusion occurs primarily between **Mistral and Meta LLaMA (`10.5%`)**, which share similar open-weight decoder architectures and instruction-tuning datasets.
3. **Right Panel — Cross-Genre Domain Generalization (`Essays`, `Fiction`, `News`):**
   - **Binary Detection (blue & green bars)** is highly stable across all domains: **`99.29%` Macro F1 in Fiction**, **`98.57%` in Essays**, and **`96.05%` in News**.
   - **23-Class Model Attribution (red bars)** reaches an extraordinary **`96.30%` exact accuracy in Fiction**, compared to `77.59%` in Essays and `66.70%` in News. **Why?** Because creative fiction gives each LLM room to express distinct narrative, dialogue, and punctuation habits, whereas hard news reporting enforces a rigid, standardized journalistic template that compresses stylistic variance across models.

### 2. Speaker Note (Ready to Deliver)
> *"Slide 7 presents our error diagnostics across three views. On the left, the binary confusion matrix confirms balanced precision and recall: 98.0% recall on human texts and 99.1% recall on AI texts. In the center, the 7-by-7 architectural family confusion matrix shows strong diagonal recall—reaching 98.0% for Human and 92.7% for OpenAI—with minor cross-family confusion occurring mainly between open-weight LLaMA and Mistral models. On the right, cross-genre evaluation demonstrates robust domain generalization: binary Macro F1 stays above 96% across all genres, while 23-class model attribution peaks at 96.3% in Fiction, where stylistic freedom amplifies each LLM's unique fingerprint, compared to 66.7% in standardized News reporting."*

---

## Slide 8 — Step 4C: Per-Author & Per-LLM Detection and Attribution

### 1. Deep Context & Subject Mastery
- **How to Read the Horizontal Breakdown (`per_llm_detection_and_attribution.png`):**
  Evaluates performance individually for all **23 candidate classes** (`human` + `22` AI models) across three metrics:
  - **Blue Bar (Task 1: Binary Detection Recall):** How reliably the detector flags each model as AI (or Human as Human). **13 out of 22 AI models have a `100.0%` binary detection rate (zero evasion!)**. Even adversarial paraphrased models (`gpt-4-turbo-paraphrase` and `gemini-pro-paraphrase`) are detected at **`98.3%` and `95.7%`**!
  - **Green Bar (Task 2: Architectural Family Attribution Accuracy — 7 Families):** How often the model predicts the correct vendor family.
  - **Red Bar (Task 2: Exact Model Attribution Accuracy — 23 Classes):** How often the model predicts the exact model version string.
- **The Key Scientific Insight ("Sibling Model Confusion"):**
  Look at the models toward the bottom where the red bar (exact model) is lower, but the green bar (vendor family) jumps much higher:
  - For **`gpt-4-turbo`**, exact 23-class accuracy is `57.1%` (red bar), but OpenAI family accuracy is **`96.8%` (green bar)**! Why? Because almost every time `gpt-4-turbo` is misclassified, it is predicted as **`gpt-4o`**—its direct sibling from the GPT-4 family!
  - Similarly, **`llama-2-70b-chat` (`49.2%` exact vs. `75.4%` family)** and **`llama-2-7b-chat` (`45.8%` exact)** are confused almost exclusively with each other (they share the exact same Meta LLaMA-2 tokenizer and pre-training corpus, differing only in parameter count).
  - Likewise, **`mistral-7b-instruct`** is primarily confused with **`mixtral-8x7b-instruct`** (which is a Mixture-of-Experts built directly from Mistral-7B weights).
  - **Takeaway:** Even when exact version attribution is challenging due to identical pre-training pipelines, our feature space accurately recovers the **true architectural genealogy** of the underlying LLM!

### 2. Speaker Note (Ready to Deliver)
> *"Finally, Slide 8 breaks down performance across each of the 23 individual candidate models. First, looking at the blue bars, binary detection recall reaches 100%—zero evasion—on 13 out of the 22 AI models, and remains above 95.7% even on paraphrased adversarial models like Gemini-Pro-Paraphrase. Second, looking at the red bars, exact 23-class model attribution exceeds 88% to 97% on distinctive models such as o3-mini (97.7%), GPT-4.5 (95.6%), GPT-4o-mini (90.7%), and LLaMA-3.3-70B (88.9%). Most importantly, comparing the red and green bars reveals that whenever exact model attribution drops—for instance on GPT-4-Turbo (57.1% exact vs. 96.8% family accuracy) or LLaMA-2-7B vs. 70B—misclassifications occur almost exclusively between sibling models of the exact same architecture family. This confirms that our feature space genuinely captures the underlying genealogical signature of each LLM family. Thank you, and I am happy to take any questions."*

---

## 9. Quick Mathematical & Conceptual Glossary for Defense Q&A

| Metric / Concept | Mathematical Definition | Why We Use It in This Project |
| :--- | :--- | :--- |
| **Macro F1-Score** | $\frac{1}{k} \sum_{c=1}^{k} F1_c$ (unweighted mean of per-class F1 scores) | Official PAN evaluation metric: treats every class equally regardless of sample size imbalance. |
| **Sublinear TF-IDF** | $\text{wf}_{t,d} = (1 + \ln \text{tf}_{t,d}) \cdot \ln\left(\frac{N}{\text{df}_t}\right)$ | Dampens term-frequency bursts in longer documents so that text length does not bias lexical weights. |
| **One-Way ANOVA $F$** | $F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$ | Quantifies how strongly a continuous feature separates the 22 unordered LLM classes relative to within-model variance. |
| **Mutual Information** | $I(X; Y) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}$ | Captures arbitrary **non-linear** statistical dependencies between a feature $X$ and output $Y$, complementing linear Pearson $r$. |
| **Permutation Importance** | $\Delta F1_j = F1_{\text{val}} - F1_{\text{val, shuffled}(X_j)}$ | Measures actual generalization reliance on feature $j$ by randomly permuting its validation column and recording the drop in Macro F1. |
| **Top-3 Accuracy (23c)** | Fraction of samples where the true class is among the top 3 highest-scoring predictions (`94.93%`) | Crucial when evaluating 23 classes that include near-identical sibling checkpoints (`gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`). |
