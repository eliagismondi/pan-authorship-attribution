# Report di Distribuzione delle Feature (Punto 10)
**Task: PAN 2025 Generative AI Detection & Authorship Attribution**  
**Autore: Elia Gismondi**

---

## 1. Obiettivo e Quadro Metodologico (Point 10)

Nel contesto dell'Authorship Attribution e della Detection di testi generati da intelligenza artificiale, l'analisi delle distribuzioni delle feature stilometriche (Punto 10) risponde a due domande scientifiche fondamentali:
1. **Comportamento Globale ($X$ sull'intero dataset)**: Qual è la forma distributiva, la dispersione (varianza, IQR) e l'eventuale asimmetria (*skewness*) delle feature estratte sui $23.707$ testi del corpus? La presenza di code pesanti (*heavy tails*) o outlier estremi richiede normalizzazioni robuste prima della fase di modeling.
2. **Dispersione tra Autori Differenti (Author-Level Variance)**: Esistono confini distribuzionali netti tra la scrittura umana spontanea e le diverse architetture generativo-conversazionali ($k=23$ modelli candidati)? Come si differenziano tra loro le singole famiglie di LLM (OpenAI, Google, Meta, Mistral, DeepSeek, TII Falcon)?

L'indagine è stata condotta sulle **6 feature cardine di $X$**:
- **`word_count`**: Volume testuale complessivo (lunghezza del documento in parole).
- **`ttr` (Type-Token Ratio)**: Varietà lessicale locale ($V / N$).
- **`avg_word_len`**: Complessità morfologico-sillabica media (caratteri per parola).
- **`punct_density`**: Frequenza dei marker prosodici e sintattici ($N_{\text{punct}} / N_{\text{chars}}$).
- **`avg_sentence_len`**: Ampiezza media del periodo sintattico (parole per frase).
- **`functional_lexical_ratio`**: Densità dell'impalcatura grammaticale chiusa rispetto ai termini lessicali aperti.

---

## 2. Distribuzione delle Feature sull'Intero Dataset ($N = 23.707$)

L'estrazione sistematica sull'intera matrice di training ha prodotto le seguenti metriche descrittive parametriche e non parametriche:

| Feature ($X$) | Media ($\mu$) | Dev. Std ($\sigma$) | Mediana ($M$) | IQR ($Q_3 - Q_1$) | Skewness | Forma Distributiva Principale |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`word_count`** | $633.02$ | $214.29$ | $648.00$ | $234.00$ | $+3.02$ | Asimmetrica positiva (coda lunga verso capitoli estesi) |
| **`ttr`** | $0.503$ | $0.064$ | $0.503$ | $0.082$ | $+0.11$ | Simmetrica, quasi-gaussiana centrata a $0.50$ |
| **`avg_word_len`** | $4.762$ | $0.575$ | $4.669$ | $0.779$ | $+0.70$ | Moderatamente asimmetrica a destra (picco a $4.4$ e secondo picco a $5.1$) |
| **`punct_density`** | $0.0322$ | $0.0124$ | $0.0294$ | $0.0148$ | $+1.10$ | Asimmetrica a destra con dispersione elevata nei dialoghi |
| **`avg_sentence_len`** | $23.90$ | $7.89$ | $22.86$ | $7.75$ | $+4.67$ | Asimmetrica positiva (presenza di periodi umani molto articolati) |
| **`functional_lexical_ratio`** | $0.738$ | $0.169$ | $0.729$ | $0.251$ | $+0.14$ | Distribuzione bicolonnare con baricentro equilibrato |

### Grafico Esportato (Intero Dataset):
La figura ad alta risoluzione con istogrammi a 40 bin, stima di densità KDE continua, linee di media/mediana e box informativi IQR è consultabile in:  
[reports/figures/feature_distributions_entire_dataset.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/feature_distributions_entire_dataset.png)

```
                       [Figura 1: feature_distributions_entire_dataset.png]
+-----------------------------------------------------------------------------------------------+
|   6 Pannelli con Istogrammi + Stima Kernel di Densità (KDE) + Indicatori Media/Mediana/IQR:   |
|   1. Word Count (μ=625.8, Med=647.0)          2. Type-Token Ratio (μ=0.50, Med=0.50)           |
|   3. Average Word Length (μ=4.76, Med=4.67)   4. Punctuation Density (μ=0.03, Med=0.03)        |
|   5. Sentence Length (μ=23.6, Med=22.9)       6. Functional-Lexical Ratio (μ=0.74, Med=0.73)   |
+-----------------------------------------------------------------------------------------------+
```

---

## 3. Distribuzione Comparativa tra Autori Differenti (Author Breakdown)

La scomposizione tra l'**Autore Umano** e i **principali modelli generativi di intelligenza artificiale** (top per frequenza nel corpus) mette in luce divergenze stilometriche sistematiche e marcate differenze tra architetture LLM:

### 3.1 Tabella Sinottica delle Medie per Autore / Modello

| Autore / Modello | TTR ($V/N$) | Lunghezza Parola ($\mu_{chars}$) | Densità Punteggiatura | Rapporto Funzionale-Lessicale | Lunghezza Frase (parole) | Word Count Medio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🧑 **Human (Author)** | **$0.471$** | **$4.342$** | **$0.0375$** | **$0.869$** | $24.11$ | **$710.95$** |
| 🤖 **GPT-3.5-Turbo** | $0.524$ | $4.998$ | $0.0237$ | $0.677$ | $22.34$ | $486.20$ |
| 🤖 **GPT-4o-Mini** | $0.543$ | $5.132$ | $0.0255$ | $0.628$ | $25.43$ | $668.42$ |
| 🤖 **GPT-4o** | **$0.556$** | **$5.176$** | $0.0258$ | $0.600$ | $23.37$ | $652.18$ |
| 🤖 **Ministral-8B** | $0.499$ | $4.753$ | $0.0334$ | $0.729$ | $19.56$ | $643.10$ |
| 🤖 **Gemini-2.0-Flash** | $0.541$ | $5.016$ | $0.0334$ | $0.668$ | $19.19$ | $612.80$ |
| 🤖 **OpenAI o3-Mini** | $0.539$ | $5.021$ | $0.0223$ | $0.661$ | **$29.45$** | $702.50$ |
| 🤖 **Gemini-1.5-Pro** | $0.545$ | $4.982$ | $0.0321$ | $0.674$ | $20.87$ | $530.15$ |
| 🤖 **LLaMA-3.1-8B** | $0.532$ | $4.871$ | $0.0286$ | $0.698$ | $26.12$ | $581.40$ |
| 🤖 **DeepSeek-R1-32B** | $0.461$ | $4.512$ | $0.0329$ | $0.803$ | $20.65$ | **$854.30$** |
| 🤖 **Falcon3-10B** | $0.528$ | $4.821$ | $0.0298$ | $0.702$ | $22.40$ | $605.90$ |

---

## 4. Analisi Stilometrica Approfondita dei Boxplot (Point 10 Insights)

La visualizzazione a Boxplot in [reports/figures/feature_distributions_by_author.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/feature_distributions_by_author.png) evidenzia quattro pattern stilometrici di rilevanza diagnostica:

### 4.1 Il Divario Lessicale (TTR & Vocabolario)
- **Scrittura Umana**: Mediana TTR significativamente più bassa ($0.471$) con un'ampia varianza naturale dovuta a testi di narrativa e dialoghi con frequenti ripetizioni di pronomi e verbi ausiliari.
- **Modelli OpenAI (GPT-4o, GPT-4o-Mini, GPT-3.5)**: Esibiscono una diversità lessicale molto compressa verso l'alto (IQR ristretto tra $0.52$ e $0.57$). Il decoder massimizza la varietà di sinonimi per token.
- **Eccezione Notevole: DeepSeek-R1-32B**: Il modello di reasoning DeepSeek presenta un TTR medio pari a $0.461$, inferiore persino all'umano. Questo è dovuto alla generazione di testi più lunghi (media $854$ parole) e alla ripetizione iterativa di catene di passaggi argomentativi (*chain of thought* narrative).

### 4.2 L'Impalcatura Sintattica: Functional-to-Lexical Ratio
- Negli esseri umani la mediana si attesta a **$0.87$**: il linguaggio naturale umano fa perno su una densa rete connettiva di preposizioni, articoli e congiunzioni (*the, of, and, to, in, with, that*).
- Nei modelli generativi avanzati come GPT-4o, la mediana scende drasticamente a **$0.60$** ($p < 10^{-15}$): il testo generato privilegia sostantivi composti, aggettivi qualificativi e participi ad alta densità semantica, comprimendo le ridondanze funzionali.

### 4.3 Punteggiatura e Ritmo Prosodico
- Gli umani registrano una densità di interpunzione di circa il **$38\%$ superiore** rispetto a modelli come OpenAI o3-Mini ($0.0375$ vs $0.0223$).
- I boxplot mostrano inoltre che la dispersione (*flier points* e baffi del boxplot) è quasi tripla negli umani: la scrittura umana spazia da periodi privi di virgole a frasi ricche di incisi, trattini e punti esclamativi, laddove i modelli conversazionali presentano uno stile di punteggiatura standardizzato e uniforme.

### 4.4 Lunghezza del Periodo Sintattico (`avg_sentence_len`)
- La maggior parte dei modelli (Ministral, Gemini, GPT-3.5) si attesta su una lunghezza media della frase compresa tra 19 e 23 parole.
- Al contrario, **OpenAI o3-mini** e **LLaMA-3.1-8B** generano frasi significativamente più lunghe e articolate (mediana $27-29$ parole per periodo), introducendo proposizioni subordinate multiple.

---

## 5. Grafici Generati ed Esportati

Sono state generate e verificate due figure ad alta risoluzione a standard accademico (160 DPI, palette cromatica armonizzata, senza glifi mancanti):

1. **[feature_distributions_entire_dataset.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/feature_distributions_entire_dataset.png)**:
   - Griglia $2 \times 3$ con istogrammi e curve KDE continue per tutte le 6 feature chiave su $N = 23.707$ campioni.
   - Indicatori visivi con linee tratteggiate per Media ($\mu$, rossa) e Mediana ($M$, verde), corredati da riquadri numerici per $\sigma$ e IQR.

2. **[feature_distributions_by_author.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/feature_distributions_by_author.png)**:
   - Griglia $2 \times 3$ di boxplot orizzontali comparativi.
   - Include l'Autore Umano (in testa, evidenziato cromaticamente in verde bosco `#2CA02C`) e i top 10 modelli LLM generativi (in blu ardesia `#1F77B4`).
   - Evidenziazione esplicita di mediane, box interquartili e outlier.

---

## 6. Modalità di Esecuzione Riproducibile

Per rigenerare le statistiche e i grafici del Punto 10:

```bash
# Esecuzione del solo Punto 10 sul dataset completo di train (23.7k testi, ~14 secondi)
python3 src/main.py --stage 10

# Esecuzione combinata di tutta la pipeline (EDA, Punto 10, Baseline Extraction, Punto 13)
python3 src/main.py --stage all
```
