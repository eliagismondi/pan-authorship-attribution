# Report su Matrice di Correlazione, Collinearità e Potere Predittivo (Punti 9, 13)
**Task: PAN 2025 Generative AI Detection & Authorship Attribution**  
**Autore: Elia Gismondi**

---

## 1. Obiettivo e Quadro Metodologico (Points 9, 13)

L'analisi della matrice di correlazione (Punti 9 e 13) risponde a due esigenze analitiche fondamentali:
1. **Analisi della Collinearità Inter-Feature (Inter-Feature Collinearity)**: Individuare coppie o cluster di variabili che misurano lo stesso costrutto stilistico o linguistico sottostante ($|r| \ge 0.60$). In modelli lineari come la Regressione Logistica o Linear SVM, variabili fortemente collineari gonfiano la varianza dei coefficienti stimati ($\text{Var}(\hat{\beta}_j) \propto \frac{1}{1 - R_j^2}$), rendendo i pesi instabili e poco interpretabili.
2. **Potere Predittivo rispetto all'Output Target ($y$)**: Quantificare la correlazione empirica lineare (coefficiente di Pearson $r$) tra ciascuna feature numerica $X_j$ e l'etichetta target binaria $y \in \{0, 1\}$ ($y=0$ Scritto da Umano, $y=1$ Generato da Intelligenza Artificiale), ordinando le feature per forza discriminativa.

L'analisi è stata eseguita sull'**intero dataset di training ($N = 23.707$ record)**.

---

## 2. Potere Predittivo delle Feature rispetto al Target Output $y$

Ordinando le feature estratte in base al loro coefficiente di correlazione con $y$, emergono due poli stilometrici netti:

| Feature Stilometrica | Pearson $r$ con $y$ | Direzione / Bias | Intensità Predittiva | Interpretazione Teorica e Forense |
| :--- | :---: | :---: | :---: | :--- |
| **`avg_word_len`** | **$+0.5755$** | **AI** 🤖 | **Forte** | Gli LLM prediligono sistematicamente termini formali, latinismi e vocaboli accademici polisillabici (*demonstrates, innovative, perspective, encapsulates*). |
| **`ttr` (Type-Token Ratio)** | **$+0.3872$** | **AI** 🤖 | **Moderata-Forte** | L'algoritmo di decodifica dei modelli AI massimizza l'entropia locale del vocabolario, evitando ripetizioni spontanee tipiche della voce umana. |
| **`honore_r`** | **$+0.2931$** | **AI** 🤖 | **Moderata** | Misura l'uso di vocaboli rari (*hapax legomena*); gli LLM campionano vocaboli da uno spazio distribuzionale molto più ampio. |
| **`ratio_word_to_sentence_len`** | **$+0.2444$** | **AI** 🤖 | **Moderata** | L'AI impacchetta parole mediamente più lunghe entro frasi di lunghezza fissa calibrata dall'istruzione di prompt. |
| **`yule_k`** | **$+0.2146$** | **AI** 🤖 | **Moderata** | Concentrazione lessicale invariante rispetto alla lunghezza. |
| **`digit_density`** | $+0.0517$ | AI 🤖 | Debole | Presenza comparabile di riferimenti numerici nei saggi comparativi. |
| **`syntactic_pause_ratio`** | $+0.0013$ | Neutro ⚖️ | Trascurabile | Il rapporto virgole/frasi complessivo è simile sui grandi volumi, ma differisce per genere. |
| **`avg_sentence_len`** | $-0.0211$ | Umano 🧑 | Trascurabile | La lunghezza media globale delle frasi non discrimina da sola Human vs AI (richiede l'interazione con la varianza o con `ratio_word_to_sentence_len`). |
| **`comma_density`** | $-0.0932$ | Umano 🧑 | Debole | Gli umani impiegano più virgole narrative. |
| **`expressive_punct_ratio`** | $-0.1487$ | Umano 🧑 | Debole-Moderata | Presenza superiore di dialoghi ed enfasi soggettiva. |
| **`quote_density`** | **$-0.2290$** | **Umano** 🧑 | **Moderata** | Dialoghi in narrativa umana e citazioni dirette nel giornalismo d'autore. |
| **`brunet_w`** | **$-0.3219$** | **Umano** 🧑 | **Moderata-Forte** | Coerente con la formulazione matematica inversa di $W$ rispetto alla diversità lessicale. |
| **`punct_density`** | **$-0.3837$** | **Umano** 🧑 | **Moderata-Forte** | La scrittura umana globale presenta una densità di interpunzione di oltre un terzo superiore rispetto all'output sintetico. |
| **`dash_density`** | **$-0.4298$** | **Umano** 🧑 | **Forte** | I trattini (em-dash `—`, en-dash `–`, trattino `-`) per incisi e precisazioni emotive sono un marker forense marcatamente umano. |
| **`functional_lexical_ratio`** | **$-0.6125$** | **Umano** 🧑 | **Molto Forte** | **Il singolo predittore più potente dell'intero corpus**: gli umani utilizzano una fitta rete connettiva funzionale (*the, of, and, in, with...*), mentre l'AI genera testo ad altissima densità semantica compressa. |

### Grafico del Potere Predittivo:
La figura esportata ad alta risoluzione illustra il ranking divergente delle feature (in verde i predittori umani, in blu i predittori AI):  
[reports/figures/feature_predictive_power_ranking.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/feature_predictive_power_ranking.png)

---

## 3. Analisi della Collinearità tra Feature (Coppie con $|r| \ge 0.60$)

L'ispezione della matrice inter-feature evidenzia le seguenti correlazioni elevate, indicando chiaramente **quali variabili misurano la stessa metrica sottostante**:

```
⚠️ Collinear Pair: brunet_w                   <-> ttr                      (r = -0.9474)
⚠️ Collinear Pair: punct_density              <-> quote_density            (r = +0.8244)
⚠️ Collinear Pair: functional_lexical_ratio   <-> avg_word_len             (r = -0.8030)
⚠️ Collinear Pair: ratio_word_to_sentence_len <-> avg_sentence_len         (r = -0.7861)
⚠️ Collinear Pair: syntactic_pause_ratio      <-> avg_sentence_len         (r = +0.7205)
⚠️ Collinear Pair: brunet_w                   <-> honore_r                 (r = -0.7029)
⚠️ Collinear Pair: expressive_punct_ratio     <-> quote_density            (r = +0.6901)
⚠️ Collinear Pair: honore_r                   <-> ttr                      (r = +0.6873)
⚠️ Collinear Pair: punct_density              <-> avg_word_len             (r = -0.6777)
```

### 3.1 Cluster 1: Ricchezza Lessicale (`ttr` $\leftrightarrow$ `brunet_w` $\leftrightarrow$ `honore_r`)
- **Correlazione**: $r = -0.947$ tra Brunet's $W$ e TTR; $r = +0.687$ tra Honoré's $R$ e TTR.
- **Spiegazione Matematica e Stilistica**: Entrambe le metriche sono formulazioni chiuse che mirano a misurare la varietà lessicale $V$ in funzione della lunghezza del testo $N$. Misurano esattamente la stessa dimensione latente.
- **Azione per la Feature Selection**: Includere simultaneamente $W$ e TTR in una regressione logistica senza penalizzazione provocherebbe multicollinearità perfetta. Si consiglia di conservare **`ttr`** per linearità e interpretabilità, oppure di applicare una regolarizzazione L1 (Lasso).

### 3.2 Cluster 2: Rapporto Morfologico-Funzionale (`avg_word_len` $\leftrightarrow$ `functional_lexical_ratio`)
- **Correlazione**: $r = -0.803$.
- **Spiegazione Stilistica**: Le parole funzionali dell'inglese (*a, an, in, on, of, to, it, he, is*) sono lemmi monosillabici o bisillabici corti (2–3 caratteri). Di conseguenza, all'aumentare della quota di parole funzionali, la lunghezza media dei caratteri per parola decresce deterministicamente.
- **Azione per la Feature Selection**: Benché misurino aspetti collegati, `functional_lexical_ratio` cattura la grammatica mentre `avg_word_len` risente del registro linguistico. Mantenere entrambe con regolarizzazione L2 (Ridge) o selezionarne una in alberi decisionali.

### 3.3 Cluster 3: Struttura Periodale (`avg_sentence_len` $\leftrightarrow$ `syntactic_pause_ratio` $\leftrightarrow$ `ratio_word_to_sentence_len`)
- **Correlazione**: $r = +0.721$ tra lunghezza frase e pause sintattiche; $r = -0.786$ tra lunghezza frase e rapporto parola/frase.
- **Spiegazione Stilistica**: Frasi con un numero elevato di parole contengono necessariamente più proposizioni coordinate e subordinate separate da virgole, incrementando di pari passo le pause sintattiche.

### 3.4 Cluster 4: Marker di Interpunzione Dialogica (`punct_density` $\leftrightarrow$ `quote_density` $\leftrightarrow$ `expressive_punct_ratio`)
- **Correlazione**: $r = +0.824$ tra densità punteggiatura totale e densità virgolette; $r = +0.690$ con la punteggiatura espressiva.
- **Spiegazione Stilistica**: Nei testi narrativi con dialoghi, ogni battuta introduce coppie di virgolette e trattini, determinando gran parte dell'incremento totale della densità di interpunzione.

---

## 4. Heatmap di Correlazione Esportata

La mappa termica ad alta risoluzione, generata sull'intero dataset ($N = 23.707$), è salvata in:  
[reports/figures/correlation_heatmap.png](file:///Users/eliagismondi/Desktop/Politecnico%20di%20Lodz/pan-authorship-attribution/reports/figures/correlation_heatmap.png)

Essa visualizza chiaramente:
- La diagonale principale identitaria ($r = 1.00$).
- I quadranti di collinearità in blu scuro (correlazione negativa, fino a $-0.95$) e rosso acceso (correlazione positiva, fino a $+0.82$).
- L'ultima riga e colonna (`target_is_ai`), che quantifica all'istante il potere predittivo di ciascuna feature rispetto all'identità dell'autore / classe AI.

---

## 5. Sintesi Operativa per lo Stage 5 (Modellazione e Classificazione)

L'analisi svolta sui Punti 9 e 13 fornisce raccomandazioni concrete per lo Stage 5:
1. **Regolarizzazione Necessaria**: Data la presenza di forti coppie collineari ($|r| \ge 0.80$), i classificatori lineari dovranno adottare penalizzazioni **ElasticNet** o **L1 (Lasso)** per azzerare i coefficienti ridondanti.
2. **Feature Core ad Alto Rendimento**: Le due colonne portanti della predizione binaria sono **`functional_lexical_ratio`** ($r = -0.613$) e **`avg_word_len`** ($r = +0.576$), seguite da **`dash_density`** ($r = -0.430$) e **`ttr`** ($r = +0.387$).
3. **Robustezza dei Modelli ad Albero**: Per modelli ensemble (Random Forest, LightGBM, XGBoost), le interazioni non lineari tra queste feature (es. ratio lunghezza parola/frase combinato con densità di punteggiatura) garantiranno un'elevata accuratezza anche in scenari cross-genre.

---

## 6. Modalità di Esecuzione Riproducibile

```bash
# Esecuzione della matrice di correlazione e del ranking predittivo (Punti 9, 13)
python3 src/main.py --stage 9

# Esecuzione completa di tutti i punti della pipeline
python3 src/main.py --stage all
```
