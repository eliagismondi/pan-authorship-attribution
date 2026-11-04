# Guida Completa alla Presentazione, Padronanza Teorica e Speaker Notes (Slide 1–8)
**Progetto:** PAN 2025 Shared Task 1 — Generative AI Detection & Authorship Attribution  
**Autore:** Elia Gismondi  
**File Slide LaTeX di riferimento:** [`reports/presentation_slides.tex`](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/presentation_slides.tex)

---

## Indice della Guida
1. [Slide 1 — Titolo, Obiettivi e Architettura Dual-Task](#slide-1--titolo-obiettivi-e-architettura-dual-task)
2. [Slide 2 — Steps 1 & 2: Multi-Level Feature Space ($D = 4,600$) & Esecuzione Corpus](#slide-2--steps-1--2-multi-level-feature-space-d--4600--esecuzione-corpus)
3. [Slide 3 — Step 3A: Feature Influence on Binary Output ($y \in \{0, 1\}$)](#slide-3--step-3a-feature-influence-on-binary-output-y-in-0-1)
4. [Slide 4 — Step 3B: Discriminating Across 22 AI Models (Multiclass ANOVA)](#slide-4--step-3b-discriminating-across-22-ai-models-multiclass-anova)
5. [Slide 5 — Step 3C: Feature Family Ablation & Leave-One-Out Impact](#slide-5--step-3c-feature-family-ablation--leave-one-out-impact)
6. [Slide 6 — Step 4A: Dual-Task Baseline Models Benchmark ($N_{\text{val}} = 3,589$)](#slide-6--step-4a-dual-task-baseline-models-benchmark-n_textval--3589)
7. [Slide 7 — Step 4B: Confusion Matrices & Cross-Domain Generalization](#slide-7--step-4b-confusion-matrices--cross-domain-generalization)
8. [Slide 8 — Step 4C: Per-Author & Per-LLM Detection and Attribution](#slide-8--step-4c-per-author--per-llm-detection-and-attribution)
9. [Glossario Matematico Rapido per l'Esposizione](#9-glossario-matematico-rapido-per-lesposizione)

---

## Slide 1 — Titolo, Obiettivi e Architettura Dual-Task

### 1. Contesto e Padronanza dell'Argomento
Questa presentazione copre i 4 step operativi richiesti dalla professoressa dopo l'analisi esplorativa iniziale (EDA):
1. **Extract features**: Trasformare i testi grezzi in vettori numerici interpretabili.
2. **Run them**: Eseguire la pipeline su tutti i **27.296 testi** del dataset PAN 2025 (`23.707` di Training + `3.589` di Validation).
3. **Measure their influence for output**: Quantificare matematicamente quanto ogni feature pesa sulla decisione finale.
4. **Run the baseline model**: Addestrare e confrontare i modelli di baseline della letteratura.

**Il valore aggiunto chiave del nostro lavoro (Dual-Task):**
Non ci siamo limitati a dire se un testo è *Umano (`0`) o AI (`1`)* (**Task 1: Binary Detection**), ma abbiamo sfruttato i metadati del corpus per risolvere anche il problema più difficile di **LLM Fingerprinting / Model Attribution (Task 2)**: identificare **quale specifico modello tra 23 candidati** (`1` classe umana + `22` modelli linguistici appartenenti a 6 grandi famiglie: OpenAI, Google Gemini, Meta LLaMA, Mistral, DeepSeek/Qwen, TII Falcon) ha generato il testo.

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"Good morning everyone. In this presentation, we cover the four core steps of our experimental pipeline for the PAN 2025 Shared Task: feature extraction, full-corpus vectorization, statistical feature influence analysis, and baseline model evaluation. Importantly, we framed our pipeline as a Dual-Task system: first, Binary Detection to separate Human from AI-generated texts across 27,296 documents; and second, fine-grained Model Attribution across 23 candidate classes, identifying which specific LLM architecture wrote each text."*

### 3. Traccia in Italiano (Cosa stai dicendo)
> *"Buongiorno a tutti. In questa presentazione mostriamo i 4 step implementati sul dataset PAN 2025 (27.296 testi): estrazione delle feature, esecuzione sull'intero corpus, misurazione dell'influenza statistica e multivariata sull'output e valutazione dei modelli baseline. Abbiamo strutturato il sistema come Dual-Task: oltre a distinguere Umano da AI (Task 1 binario), prediciamo anche quale specifico modello tra i 22 LLM candidati ha generato il testo (Task 2 multiclasse)."*

---

## Slide 2 — Steps 1 & 2: Multi-Level Feature Space ($D = 4,600$) & Esecuzione Corpus

### 1. Contesto Teorico e Tecnico Profondo
- **Perché un approccio "Multi-Level" a 4 famiglie?**
  Nella letteratura sull'Authorship Attribution (Stamatatos, 2009), usare solo le parole del testo rischia di confondere lo **stile dell'autore** con l'**argomento (topic)** del testo (*topic confounding*). Per evitarlo, il nostro vettore di **4.600 dimensioni** combina 4 livelli complementari:
  1. **Stilometria e Ricchezza Lessicale (`30` feature dense)**: Cattura il ritmo macro-strutturale e la diversità del vocabolario. Include metriche classiche (`ttr`, lunghezze frasi/parole, 9 densità di punteggiatura specifiche) e indici matematici invarianti alla lunghezza del testo come **Yule's $K$**, **Brunet's $W$** ($W = N^{V^{-0.172}}$) e **Honoré's $R$** ($R = \frac{100 \ln N}{1 - V_1/V}$, che misura la ricchezza di *hapax legomena*, cioè parole usate una sola volta).
  2. **Function Words (`70` feature dense)**: Preposizioni, articoli, ausiliari, congiunzioni (*the, of, and, to, so, was...*). Sono **indipendenti dal tema**: che un autore scriva di sport, politica o narrativa, usa le preposizioni e i connettivi in modo inconscio e costante.
  3. **Char N-Grams (`3.000` feature sparse TF-IDF)**: Sequenze di 3–4 caratteri con confini di parola (`analyzer='char_wb'`). Catturano prefissi, suffissi morfologici (`-ing`, `-tion`, `-ly`) e combinazioni di punteggiatura. Usiamo `sublinear_tf=True` (sostituisce la frequenza grezza $tf$ con $1 + \ln(tf)$) per evitare che parole ripetute in testi lunghi dominino il vettore.
  4. **Word N-Grams (`1.500` feature sparse TF-IDF)**: Unigrammi e bigrammi lessicali preservando le stop-words per catturare brevi costrutti sintattici (*"said the"*, *"as well"*).
- **Prevenzione assoluta del Data Leakage:**
  Lo `StandardScaler` (che porta le 100 feature dense a media 0 e deviazione standard 1) e i dizionari `TfidfVectorizer` vengono calcolati (`fit`) **esclusivamente sui 23.707 testi di Train** e poi applicati (`transform`) ai 3.589 testi di Validation.
- **Lettura del Grafico (`stylometric_distributions.png`)**:
  Mostra le curve di densità (KDE) delle 3 feature stilometriche cardine:
  - *Sinistra (TTR)*: La curva blu (AI) è spostata a destra rispetto alla verde (Human) $\to$ gli LLM hanno una varietà lessicale locale più alta perché il loro *repetition penalty* durante il decoding evita di ripetere la stessa parola.
  - *Centro (Punctuation Density)*: La curva verde (Human) è più alta e spostata a destra $\to$ gli umani usano il ~32% di punteggiatura in più (dialoghi, trattini, punto e virgola).
  - *Destra (Average Word Length)*: La curva blu (AI) è nettamente spostata verso parole più lunghe (`5.02` vs `4.34` caratteri) $\to$ stile accademico/formale degli LLM.

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"In Steps 1 and 2, we engineered a 4,600-dimensional feature space combining four topic-invariant families: 30 dense stylometric and lexical richness indices, 70 closed-class function words, 3,000 sub-word character 3-to-4-grams, and 1,500 word 1-to-2-grams with sublinear TF-IDF scaling. To strictly prevent data leakage, all scalers and vocabularies were fitted exclusively on the 23,707 training texts and applied to the 3,589 validation texts. The extraction runs at over 415 documents per second with zero NaN or infinite values. As shown in the KDE plots on the right, AI texts exhibit systematically higher lexical diversity (TTR) and longer average word lengths, whereas human authors use significantly denser punctuation."*

### 3. Possibili Domande della Professoressa (Q&A)
- **D: Perché avete applicato `StandardScaler` solo alle prime 100 feature e non ai 4.500 N-grammi?**
  - *R:* Perché le 100 feature stilometriche hanno scale eterogenee (es. `word_count` arriva a 4.000 mentre `comma_density` è `0.01`) e senza standardizzazione dominerebbero i modelli lineari; al contrario, le matrici TF-IDF sono già normalizzate con norma Euclidea $L_2 = 1$ e centrarle sottraendo la media distruggerebbe la sparsità della matrice CSR (passando da 548 MB a oltre 1 GB di RAM).

---

## Slide 3 — Step 3A: Feature Influence on Binary Output ($y \in \{0, 1\}$)

### 1. Contesto Teorico e Tecnico Profondo
- **Cosa abbiamo misurato?**
  La professoressa ha chiesto di misurare l'influenza delle feature sull'output. Lo abbiamo fatto su **due livelli complementari**:
  1. **Influenza Univariata (Pannello di Sinistra — Pearson $r$, ANOVA $F$, Mutual Information)**:
     Valuta ogni singola feature presa da sola rispetto a $y$ (`0` = Human, `1` = AI):
     - Il predittore singolo più forte in assoluto è **`functional_lexical_ratio` ($r = -0.613$, ANOVA $F = 14.234$)**: misura il rapporto tra parole grammaticali funzionali e parole di contenuto. È negativo (verde = Human) perché il linguaggio umano spontaneo usa molta più "impalcatura" grammaticale, mentre l'AI condensa informazioni lessicali dense.
     - Al secondo posto c'è **`semicolon_density` ($r = -0.582$, Mutual Info $= 0.277$)**: i modelli AI quasi non usano mai il punto e virgola se non esplicitamente istruiti nel prompt, mentre gli scrittori umani lo usano regolarmente.
     - Sul fronte AI (blu, $r > 0$), dominano **`avg_word_len` ($r = +0.576$)** e **`word_len_std` ($r = +0.562$)**.
  2. **Influenza Multivariata (Pannello di Destra — Pesi $\beta_j$ di Logistic Regression L2 su tutte le 4.600 feature + Permutation Importance)**:
     Qui tutte le 4.600 feature lavorano insieme.
     - **Permutation Importance**: Abbiamo misurato di quanto crolla il Macro F1 sul Validation set se mescoliamo casualmente una colonna: rimescolare `avg_word_len` fa perdere ben il **14,1% di Macro F1**!
     - **Scoperta linguistica sugli N-grammi**: I pesi $\beta$ positivi più alti appartengono a parole tipiche del fine-tuning RLHF dei modelli AI: ***"significant"* ($\beta = +2.03$), *"ongoing"* ($\beta = +1.54$), *"impact"* ($\beta = +1.53$), *"despite"* ($\beta = +1.46$), *"crucial"* ($\beta = +1.42$)**. Al contrario, connettivi causali e narrativi umani come ***"because"* ($\beta = -2.09$), *"although"* ($\beta = -1.83$), *"said the"* ($\beta = -1.49$)** spingono fortemente verso la classe Umana.

> **Dettaglio da Pro (se la professoressa nota il segno di `functional_lexical_ratio` nei due grafici):**
> Nel grafico di sinistra (univariato) `functional_lexical_ratio` è negativo ($r = -0.613$, Human), mentre nel grafico di destra (multivariato) ha peso $\beta = +2.67$. Perché? Perché nel modello multivariato sono presenti contemporaneamente anche tutte le **70 singole function words** (`the`, `and`, `of`, `is`, `was`), che hanno tutte pesi $\beta$ fortemente negativi (tra `-1.47` e `-0.90`). Poiché `functional_lexical_ratio` è la somma di quelle stesse parole funzionali ed è collineare con `avg_word_len` ($r = -0.82$), la regressione lineare la usa come **variabile di bilanciamento (suppressor variable)** per non contare due volte le 70 function words!

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"In Step 3A, we quantified feature influence on the binary target using both univariate statistics on the left and multivariate regularized Logistic Regression weights on the right. Univariately, the strongest human markers are functional-to-lexical ratio ($r = -0.613$) and semicolon density ($r = -0.582$), while average word length ($r = +0.576$) is the primary AI marker—causing a 14.1% drop in Macro F1 under permutation importance. On the right, inspecting the global 4,600-dimensional model reveals classic RLHF vocabulary biases: words like 'significant', 'ongoing', 'impact', 'despite', and 'crucial' carry the highest positive weights toward AI, whereas causal connectors like 'because' and 'although' strongly predict human authorship."*

---

## Slide 4 — Step 3B: Discriminating Across 22 AI Models (Multiclass ANOVA)

### 1. Contesto Teorico e Tecnico Profondo
- **Come si misura l'influenza delle feature quando ci sono 22 modelli AI diversi?**
  La correlazione di Pearson funziona solo per due classi ($0$ vs $1$). Per 22 classi AI non ordinate, il test statistico rigoroso è l'**ANOVA a una via ($F$-test multiclasse)**:
  $$F = \frac{\text{Varianza Between-Models (tra i 22 LLM)}}{\text{Varianza Within-Model (dentro lo stesso LLM)}}$$
  Più alto è $F$, più quella feature assume valori ben distinti e riconoscibili da un modello AI all'altro!
- **Cosa rivela la Heatmap (`llm_family_stylometric_fingerprints.png`)?**
  Ogni colonna rappresenta una delle 7 famiglie architetturali e ogni cella mostra lo **Z-score medio** (quante deviazioni standard quella famiglia si discosta dalla media del dataset: rosso = sopra la media, blu = sotto la media):
  1. **`uppercase_ratio` ($F = 700$)**: È il primo discriminatore tra LLM! Mistral (`+0.33z`) e Google Gemini (`+0.21z`) usano molte più maiuscole (titoli, elenchi, acronimi) rispetto a Falcon (`-0.49z`) e OpenAI (`-0.35z`).
  2. **Famiglia OpenAI (`gpt-4o`, `o3-mini`, `gpt-3.5`)**: Ha il vocabolario più ricco e vario in assoluto (`ttr = +0.65z`, `hapax_ratio = +0.58z`, `brunet_w = -0.63z`) ma usa pochissime virgole (`comma_density = -0.50z`), preferendo periodi diretti e scorrevoli.
  3. **Famiglia DeepSeek / Qwen (`deepseek-r1-distill-qwen-32b`)**: Essendo modelli orientati al *reasoning* passo-passo, producono testi strutturalmente molto più lunghi (`sentence_count = +0.80z`, `char_count = +0.61z`, `word_count = +0.60z`) ma più ripetitivi nei termini chiave (`ttr = -0.46z`).
  4. **Famiglia TII Falcon (`falcon3-10b`)**: Presenta il picco massimo di parole uniche usate una sola volta (`hapax_ratio = +0.73z` e `honore_r = +0.65z`).

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"Moving to Step 3B, we asked: which features allow us to distinguish between the 22 different AI models? Using multiclass one-way ANOVA across the AI corpus, we ranked features by their between-model to within-model variance ratio. Uppercase ratio ($F = 700$) and lexical richness indices—Root TTR, Hapax ratio, and Brunet's W—emerge as the strongest LLM fingerprints. The standardized Z-score heatmap on the right clearly illustrates each vendor's signature: OpenAI models maximize vocabulary diversity (TTR $+0.65z$) while minimizing comma density ($-0.50z$); DeepSeek reasoning models produce significantly longer, multi-sentence outputs ($+0.80z$); and Falcon exhibits the highest hapax legomena ratio ($+0.73z$)."*

---

## Slide 5 — Step 3C: Feature Family Ablation & Leave-One-Out Impact

### 1. Contesto Teorico e Tecnico Profondo
- **Cos'è un Ablation Study e perché è fondamentale per la professoressa?**
  Per dimostrare che tutte le 4 famiglie di feature sono utili, abbiamo eseguito due esperimenti sul Validation Set ($N = 3.589$):
  1. **Pannello di Sinistra (Single vs. Combined Families)**: Addestriamo il modello usando una sola famiglia alla volta e poi combinandole:
     - **Risultato straordinario sulla Stilometria**: Con **sole 30 feature stilometriche**, il modello ottiene già il **93,61% di Macro F1 binario (97,91% ROC-AUC)**! Questo dimostra che le nostre 30 feature matematiche catturano l'essenza strutturale della differenza Umano/AI senza nemmeno guardare quali parole vengono usate.
     - Tuttavia, per distinguere tra **23 modelli specifici (barre rosse)**, 30 feature non bastano (`42,0%` F1): aggiungendo i Character e Word N-grams si sale dal `42,0%` al **`73,27%` di Macro F1**.
  2. **Pannello di Destra (Leave-One-Family-Out — LOFO)**: Partiamo dal sistema completo (`4.600` feature) e togliamo una famiglia alla volta per vedere quanto perdiamo:
     - Se togliamo la **Stilometria (`LOFO: w/o Stylometry`)**, il Macro F1 sull'attribuzione dei 23 modelli crolla di **`-4,76` punti percentuali** (barra arancione a sinistra)!
     - Se togliamo i **Char N-Grams (`LOFO: w/o Char N-Grams`)**, perdiamo **`-3,06` punti percentuali**.
     - Se togliamo i **Word N-Grams**, perdiamo **`-1,58` punti percentuali**.
     - Questo dimostra matematicamente che **Stilometria globale** (lunghezze, punteggiatura, TTR) e **N-grammi locali** sono fortemente sinergici: nessuno dei due da solo raggiunge il `73,27%` del sistema combinato.

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"In Step 3C, we conducted a Feature Family Ablation study to measure the unique contribution of each feature group on both tasks. On the left, remarkably, using only our 30 dense stylometric features achieves 93.61% Macro F1 on binary detection. However, for 23-class model attribution—shown in red—combining dense stylometrics with character and word N-grams is essential, boosting Macro F1 from 42.0% up to 73.27%. On the right, our Leave-One-Family-Out analysis proves this synergy: dropping the 30 stylometric features from the full pipeline causes the sharpest drop in 23-class attribution (-4.76 percentage points), followed by character N-grams (-3.06 points)."*

---

## Slide 6 — Step 4A: Dual-Task Baseline Models Benchmark ($N_{\text{val}} = 3,589$)

### 1. Contesto Teorico e Tecnico Profondo
- **Quali modelli di baseline abbiamo confrontato e perché?**
  1. **Dummy Classifier (Majority Class)**: Predice sempre la classe più frequente (`AI` nel binario, `human` nel multiclasse a 23 classi). Serve come pavimento minimo di riferimento (`39,2%` Macro F1 binario, `2,3%` Macro F1 su 23 classi).
  2. **Complement Naive Bayes (`ComplementNB`)**: Evoluzione del Multinomial Naive Bayes progettata per dataset con classi sbilanciate. Richiede feature non-negative (abbiamo applicato `MinMaxScaler [0,1]` sulle feature dense). Raggiunge il **`90,2%` di Macro F1 binario** in appena `0,04` secondi, ma soffre sul multiclasse a 23 classi (`49,9%` Top-1 Acc) perché l'ipotesi "Naive" di indipendenza condizionata tra feature viene violata dalle forti correlazioni tra N-grammi sovrapposti.
  3. **Logistic Regression (L1 Lasso & L2 Ridge)**: Modelli lineari discriminativi che ottimizzano la Cross-Entropy regolarizzata. Ottengono **`97,8%` (L1) e `98,0%` (L2) di Macro F1 binario** (`99,71%` ROC-AUC) e **`81,6%` di Top-1 Accuracy (`94,6%` Top-3)** sulle 23 classi.
  4. **Linear SVM (`LinearSVC` — Stamatatos Standard)**: Massimizza il margine geometrico dell'iperpiano separatore in spazi ad alta dimensionalità (`4.600` dim). È il modello vincitore su tutta la linea:
     - **Task 1 Binario:** **`98,88%` in 5-Fold CV** sul Train e **`98,60%` Macro F1 (`98,72%` Accuracy, `99,84%` ROC-AUC)** sul Validation Set, con soli **26 Falsi Positivi** e **20 Falsi Negativi** su 3.589 testi!
     - **Task 2 Model Attribution (23 classi):** **`84,12%` Top-1 Accuracy**, **`94,93%` Top-3 Accuracy** (95 volte su 100 il modello esatto è tra i primi 3 predetti!), **`73,27%` Macro F1** e **`89,05%` di accuratezza sulle 7 famiglie di vendor**.
  5. **Two-Stage Hierarchical SVM (`Stage 1 Binary SVM -> Stage 2 22-LLM Specialist SVM`)**:
     Invece di un unico classificatore a 23 classi, usa il Binary SVM allo Stadio 1 per separare Umano e AI, e passa i testi AI a uno **Stage 2 SVM addestrato esclusivamente sui 22 modelli AI**. Risultato: ottiene la **migliore accuratezza esatta sui 22 modelli AI (`76,08%`)** e il **miglior Macro F1 sulle 7 famiglie (`83,14%`)**.

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"In Step 4A, we benchmarked our baseline classifiers on both tasks using 5-fold cross-validation on Train and evaluation on the 3,589 validation texts. For Binary Detection, while Complement Naive Bayes reaches 90.2% Macro F1, discriminative linear models—L1/L2 Logistic Regression and Linear SVM—exceed 98% Macro F1. Specifically, Linear SVM achieves the best binary performance with 98.60% Validation Macro F1 and 99.84% ROC-AUC, making only 46 total errors out of 3,589 texts. For Task 2—predicting which of the 23 authors or AI models wrote the text—Linear SVM and our Two-Stage Hierarchical SVM achieve 84.1% Top-1 accuracy, nearly 95% Top-3 accuracy, and 89.1% accuracy at identifying the LLM vendor family."*

---

## Slide 7 — Step 4B: Confusion Matrices & Cross-Domain Generalization

### 1. Contesto Teorico e Tecnico Profondo
Questa slide mostra **3 diagnostiche fondamentali** (da sinistra a destra nel grafico):
1. **Pannello 1 (Sinistra) — Matrice di Confusione Binaria 2×2 (Linear SVM)**:
   - Su `1.277` testi umani reali, `1.251` sono classificati correttamente (**`98,0%` Specificità / True Negative Rate**) e solo `26` sono falsi positivi (`2,0%`).
   - Su `2.312` testi generati da AI, `2.292` sono intercettati correttamente (**`99,1%` Sensibilità / Recall**) e solo `20` sfuggono come falsi negativi (`0,9%`).
2. **Pannello 2 (Centro) — Matrice di Confusione 7×7 per Famiglia Architetturale**:
   - Mostra la diagonale principale fortissima: **Human `98,0%`**, **OpenAI `92,7%`**, **Google Gemini `81,3%`**, **DeepSeek/Qwen `81,1%`**, **Falcon `78,8%`**, **Meta LLaMA `77,4%`**, **Mistral `71,5%`**.
   - Quando un modello open-weight viene confuso con un'altra famiglia, avviene quasi sempre tra **Mistral e LLaMA (`10,5%`)**, che condividono un'architettura decoder-only e ricette di pre-training molto simili.
3. **Pannello 3 (Destra) — Generalizzazione Cross-Domain sui 3 Generi (`Essays`, `Fiction`, `News`)**:
   - Il rilevamento binario (barre blu e verdi) è granitico su tutti i generi: **`99,29%` in Fiction**, **`98,57%` in Essays**, **`96,05%` in News**.
   - L'attribuzione esatta tra i 23 modelli (barra rossa) tocca il **`96,30%` nella Narrativa (`Fiction`)**, scende a `77,59%` nei Saggi (`Essays`) e a `66,70%` nel Giornalismo (`News`). **Perché?** Perché nella narrativa ogni LLM mostra tic creativi e dialogici inconfondibili, mentre negli articoli di cronaca (`News`) il formato giornalistico rigido costringe tutti i 22 modelli AI a scrivere con uno stile asciutto e uniforme.

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"Slide 7 presents our error diagnostics across three views. On the left, the binary confusion matrix confirms balanced precision and recall: 98.0% recall on human texts and 99.1% recall on AI texts. In the center, the 7-by-7 architectural family confusion matrix shows strong diagonal recall—reaching 98.0% for Human and 92.7% for OpenAI—with minor cross-family confusion occurring mainly between open-weight LLaMA and Mistral models. On the right, cross-genre evaluation demonstrates robust domain generalization: binary Macro F1 stays above 96% across all genres, while 23-class model attribution peaks at 96.3% in Fiction, where stylistic freedom amplifies each LLM's unique fingerprint, compared to 66.7% in standardized News reporting."*

---

## Slide 8 — Step 4C: Per-Author & Per-LLM Detection and Attribution

### 1. Contesto Teorico e Tecnico Profondo
- **Cosa mostra il grafico a barre orizzontali (`per_llm_detection_and_attribution.png`)?**
  Scorpora le prestazioni su **ciascuno dei 23 candidati** del Validation Set, confrontando 3 metriche per ogni riga:
  - **Barra Blu (Task 1: Binary Detection Recall)**: Quante volte il sistema capisce che quel modello è un'AI (o che `human` è umano). **Su 13 dei 22 modelli AI, il recall binario è del `100,0%` (zero evasione!)**. Anche i modelli parafrasati (`gpt-4-turbo-paraphrase` e `gemini-pro-paraphrase`, progettati appositamente per eludere i detector) vengono intercettati rispettivamente al **`98,3%` e `95,7%`**!
  - **Barra Verde (Task 2: Architectural Family Attribution Acc)**: Quante volte indoviniamo la famiglia/casa produttrice del modello.
  - **Barra Rossa (Task 2: Exact Model Attribution Acc — 23 classi)**: Quante volte indoviniamo il nome e la versione esatta del modello.
- **L'intuizione scientifica più bella da sottolineare alla professoressa ("Sibling Confusion"):**
  Guarda i modelli in fondo al grafico dove la barra rossa (modello esatto) è più bassa ma la barra verde (famiglia) resta altissima, ad esempio **`gpt-4-turbo`**:
  - L'attribuzione del modello esatto è `57,1%` (barra rossa), ma l'attribuzione della famiglia OpenAI è **`96,8%` (barra verde)**! Perché? Perché nel restante `40%` dei casi il classificatore predice **`gpt-4o`**, che è letteralmente il fratello della stessa serie GPT-4!
  - Lo stesso identico fenomeno accade tra **`llama-2-70b-chat` e `llama-2-7b-chat`**: si confondono quasi esclusivamente tra loro due (stesso identico pre-training Meta LLaMA-2, cambia solo la dimensione dei parametri da 7B a 70B), e tra **`mistral-7b` e `mixtral-8x7b`** (dove Mixtral è un Mixture-of-Experts di Mistral-7B).
  - Questo dimostra che lo spazio stilometrico + N-grammi che abbiamo costruito cattura la **vera genealogia architetturale dei modelli linguistici**!

### 2. Speaker Note (Inglese — Pronta da Esporre)
> *"Finally, Slide 8 breaks down performance across each of the 23 individual candidate models. First, looking at the blue bars, binary detection recall reaches 100%—zero evasion—on 13 out of the 22 AI models, and remains above 95.7% even on paraphrased adversarial models like Gemini-Pro-Paraphrase. Second, looking at the red bars, exact 23-class model attribution exceeds 88% to 97% on distinctive models such as o3-mini (97.7%), GPT-4.5 (95.6%), GPT-4o-mini (90.7%), and LLaMA-3.3-70B (88.9%). Most importantly, comparing the red and green bars reveals that whenever exact model attribution drops—for instance on GPT-4-Turbo (57.1% exact vs. 96.8% family accuracy) or LLaMA-2-7B vs. 70B—misclassifications occur almost exclusively between sibling models of the exact same architecture family. This confirms that our feature space genuinely captures the underlying genealogical signature of each LLM family. Thank you, and I am happy to take any questions."*

---

## 9. Glossario Matematico Rapido per l'Esposizione

Se durante la discussione la professoressa ti chiede la definizione rapida di una metrica presente nelle slide:

| Metrica / Termine | Formula / Definizione Sintetica | Perché la usiamo nel progetto |
| :--- | :--- | :--- |
| **Macro F1-Score** | $\frac{1}{k} \sum_{c=1}^{k} F1_c$ (media aritmetica non pesata degli F1 di tutte le classi) | È la metrica ufficiale PAN: dà lo stesso peso a ogni classe, anche a quelle con meno campioni nel dataset. |
| **Sublinear TF-IDF** | $\text{wf}_{t,d} = (1 + \ln \text{tf}_{t,d}) \cdot \ln\left(\frac{N}{\text{df}_t}\right)$ | Smorza l'effetto della lunghezza del testo: una parola ripetuta 20 volte pesa $1+\ln(20) \approx 4$, non 20 volte tanto. |
| **ANOVA $F$-Statistic** | $F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$ | Misura quanto una feature separa tra loro le 22 classi AI rispetto alla varianza interna a ciascun modello. |
| **Mutual Information** | $I(X; Y) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}$ | Cattura qualsiasi dipendenza statistica (anche **non-lineare**) tra una feature e il target $y$, a differenza di Pearson $r$ che vede solo relazioni lineari. |
| **Permutation Importance** | $\Delta F1_j = F1_{\text{orig}} - F1_{\text{shuffled}(X_j)}$ | Misura quanto il modello dipende realmente dalla feature $j$ sul Validation Set rimescolandone i valori e osservando il calo di Macro F1. |
| **Top-3 Accuracy (23c)** | Percentuale di volte in cui la classe vera è tra le prime 3 più probabili predette dal modello (`94,93%`) | Fondamentale quando ci sono 23 classi con modelli "fratelli" quasi identici (es. `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`). |
