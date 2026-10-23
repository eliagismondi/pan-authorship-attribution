# Report di Estrazione Baseline nello Spazio Vettoriale (Punti 6, 7)
**Task: PAN 2025 Generative AI Detection & Authorship Attribution**  
**Autore: Elia Gismondi**

---

## 1. Obiettivo e Quadro Metodologico (Points 6, 7)

In conformità con quanto definito nella Literature Review (Stamatatos 2009; Koppel et al. 2011; Barlas & Stamatatos 2020), il task di **Baseline Extraction** richiede di trasformare i testi grezzi non strutturati in uno **spazio vettoriale numerico $\mathbb{R}^{N \times D}$**, estraendo le famiglie di feature stilometriche e lessicali invarianti rispetto al tema.

La pipeline implementata in [src/feature_extraction.py](file:///Users/eliagismondi/Desktop/pan-authorship-attribution/src/feature_extraction.py) estrae simultaneamente **4 famiglie complementari di feature**, combinandole in una matrice numerica coerente con tracciamento nominale di ogni singola dimensione.

```mermaid
flowchart TD
    A["Raw Text String"] --> B["1. Stylometric Feature Extractor<br/>(TTR, Yule K, Simpson D, Punctuation, Lengths)<br/>24 Dimensioni Dense"]
    A --> C["2. Function Word Extractor<br/>(Relative frequencies of 70 closed-class terms)<br/>70 Dimensioni Dense"]
    A --> D["3. Sub-word Character N-Grams (TF-IDF)<br/>(char_wb, n=3..4, sublinear TF)<br/>3,000 Dimensioni Sparse"]
    A --> E["4. Word N-Grams (TF-IDF)<br/>(word unigrams + bigrams with function words)<br/>1,500 Dimensioni Sparse"]
    B --> F["StandardScaler (Dense Normalization)"]
    C --> F
    F --> G["scipy.sparse.hstack"]
    D --> G
    E --> G
    G --> H["Unified Feature Matrix X (Shape: N x 4,594)"]
```

---

## 2. Tassonomia delle Feature Estratte

### 2.1 Famiglia 1: Stilometria e Ricchezza Lessicale (24 Dimensioni)
Implementata nella classe `StylometricFeatureExtractor` (compatibile con Scikit-learn):
1. **Type-Token Ratio (TTR)**: $V / N$ (varietà lessicale per testo).
2. **Guiraud's R (Root TTR)**: $V / \sqrt{N}$ (compensazione parziale per testi più lunghi).
3. **Indice di Yule ($K$)**: 
   $$K = 10^4 \cdot \frac{\sum_{m} m^2 V(m, N) - N}{N^2}$$
   Misura la concentrazione del vocabolario, rigorosamente **indipendente dalla lunghezza del testo** (fondamentale per attenuare il divario tra testi umani e AI).
4. **Simpson's $D$**: Probabilità che due lemmi scelti casualmente coincidano.
5. **Hapax Legomena Ratio**: Quota di vocaboli usati una sola volta ($V_1 / V$).
6. **Dislegomena Ratio**: Quota di vocaboli usati esattamente due volte ($V_2 / V$).
7. **Statistiche sulle Parole**: Lunghezza media dei caratteri e deviazione standard ($\mu_{len}, \sigma_{len}$).
8. **Statistiche sulle Frasi**: Lunghezza media della frase in parole e conteggio frasi stimato.
9. **Marker Strutturali**: Densità di cifre numeriche (`digit_density`) e percentuale di caratteri maiuscoli (`uppercase_ratio`).
10. **Profilo della Punteggiatura**:
    - Densità totale (`punct_density = n_punct / n_chars`).
    - Densità specifica per 9 segni di interpunzione distinti: virgola (`,`), punto (`.`), punto e virgola (`;`), due punti (`:`), punto esclamativo (`!`), punto interrogativo (`?`), trattini (`-`, `—`, `–`), virgolette (`"`, `'`, `“`, `”`), parentesi (`(`, `)`, `[`, `]`).

### 2.2 Famiglia 2: Parole Funzionali / Stop-Words Chiuse (70 Dimensioni)
Implementata nella classe `FunctionWordExtractor`:
- Estrazione della frequenza relativa di **70 parole funzionali chiuse** (preposizioni, ausiliari, congiunzioni, pronomi: *the, of, and, to, a, in, that, is, was, he, for, it, with, as, his, on, be, at, by, i, this, had, not, are, but, from, or, have, an, they, which, one, you, were, her, all, she, there, would, their, we, him, been, has, when, who, will, more, no, if, out, so, said, what, up, its, about, into, than, them, can, only, other, some, could, time, these, two, may, then*).
- Normalizzazione per la lunghezza del documento ($N_{words}$) per garantire la **totale invarianza rispetto al topic (cross-topic stability)**.

### 2.3 Famiglia 3: $N$-grammi di Caratteri Sub-word (3.000 Dimensioni)
- Configurazione: `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=3000, sublinear_tf=True)`.
- L'analizzatore `char_wb` include i delimitatori di parola, consentendo di catturare prefissi, suffissi morfologici (-ing, -ed, -tion), combinazioni di punteggiatura e radici lessicali.

### 2.4 Famiglia 4: $N$-grammi di Parole Lessicali (1.500 Dimensioni)
- Configurazione: `TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1500, sublinear_tf=True)`.
- Mantiene intatte le stop-words, catturando collocazioni sintattiche tipiche (*of the, in a, it was, will be*).

---

## 3. Metriche dello Spazio Vettoriale Risultante

Il benchmark eseguito sul corpus reale dimostra la scalabilità e le proprietà algebriche dello spazio vettoriale:

| Proprietà Vettoriale | Valore Riscontrato | Note Tecniche |
| :--- | :---: | :--- |
| **Dimensioni Totali ($D$)** | **4.594 colonne** | 24 stilometriche + 70 parole funzionali + 3.000 n-grammi char + 1.500 n-grammi word |
| **Formato Matrice** | `scipy.sparse.csr_matrix` | Compressed Sparse Row per ottimizzazione RAM e calcolo parallelo |
| **Sparsità della Matrice** | **66.31%** | Equilibrata: le feature stilometriche e funzionali sono dense; gli n-grammi sono sparsi |
| **Throughput di Estrazione** | **~300 testi/secondo** | 1.000 testi processati in appena **3.35 secondi** su singola CPU |
| **Scalabilità sull'Intero Train (23.7k)** | **~75 secondi stimati** | Pienamente eseguibile in memoria senza memory leak o overhead |

---

## 4. Codice Minimo per Utilizzare la Pipeline

È sufficiente importare `create_baseline_pipeline` da [src/feature_extraction.py](file:///Users/eliagismondi/Desktop/pan-authorship-attribution/src/feature_extraction.py):

```python
from src.feature_extraction import create_baseline_pipeline

# 1. Inizializzazione della pipeline conforme a Scikit-Learn
pipeline = create_baseline_pipeline(char_max_features=3000, word_max_features=1500)

# 2. Fit & Transform sui testi grezzi
X = pipeline.fit_transform(raw_texts)  # Shape: (N, 4594)

# 3. Tracciamento nominale delle feature
feature_names = pipeline.get_feature_names_out()
print(f"Dimensione matrice: {X.shape}")
print(f"Prime 5 feature: {feature_names[:5]}")
```

Per l'esecuzione da riga di comando:
```bash
python3 src/main.py --stage 2 --records 1000
```
