# Report di Derivazione Avanzata e Analisi di Correlazione (Punto 13)
**Task: PAN 2025 Generative AI Detection & Authorship Attribution**  
**Autore: Elia Gismondi**

---

## 1. Definizione delle Feature Derivate Custom (Point 13)

Per superare i limiti delle sole metriche di superficie (conteggio parole o caratteri) e catturare la struttura sintattico-stilistica profonda dell'autore, sono state introdotte **6 nuove feature derivate custom**:

### 1.1 Rapporto Lunghezza Parola / Lunghezza Frase (`ratio_word_to_sentence_len`)
$$R_{ws} = \frac{\mu_{\text{word\_length}}}{\mu_{\text{sentence\_length}}} = \frac{\text{caratteri totali} / N_{\text{words}}}{N_{\text{words}} / N_{\text{sentences}}}$$
- **Significato Stilistico**: Misura la densità d'informazione per unità sintattica. Un valore elevato indica frasi brevi ma dense di lemmi polisillabici e tecnici; un valore basso indica frasi lunghe con parole semplici e colloquiali.

### 1.2 Densità della Punteggiatura Globale (`punct_density`)
$$\text{Punct\_Density} = \frac{\sum \text{caratteri di interpunzione}}{\text{totale caratteri}}$$
- **Significato Stilistico**: Esprime il ritmo prosodico della scrittura e la frequenza delle pause sintattiche.

### 1.3 Rapporto di Pause Sintattiche (`syntactic_pause_ratio`)
$$\text{Pause\_Ratio} = \frac{N_{\text{virgole}}}{N_{\text{frasi}}}$$
- **Significato Stilistico**: Misura il grado di subordinazione (ipotassi vs paratassi). Quantifica quanti incisi e proposizioni coordinate/subordinate l'autore inserisce in media per periodo.

### 1.4 Rapporto di Punteggiatura Espressiva (`expressive_punct_ratio`)
$$\text{Exp\_Punct} = \frac{n_{\text{exclamation}} + n_{\text{question}} + n_{\text{dash}} + n_{\text{quotes}}}{N_{\text{interpunzione\_totale}}}$$
- **Significato Stilistico**: Isola la punteggiatura non neutrale (dialoghi, enfasi, incisi a trattino) rispetto ai terminatori standard (punti e virgole).

### 1.5 Rapporto Funzionale-Lessicale (`functional_lexical_ratio`)
$$\text{FLR} = \frac{N_{\text{function\_words}}}{N_{\text{words}} - N_{\text{function\_words}}}$$
- **Significato Stilistico**: Rapporto tra parole grammaticali chiuse (*the, of, and, in, with...*) e parole di contenuto semantico aperto. Negli umani il linguaggio naturale spontaneo richiede una forte impalcatura funzionale; i modelli generativi tendono a produrre testo ad alta densità lessicale/informativa.

### 1.6 Indici Avanzati di Ricchezza Lessicale: Brunet's $W$ e Honoré's $R$
- **Brunet's $W$**: $W = N^{V^{-0.172}}$ (misura di ricchezza lessicale costante al variare di $N$).
- **Honoré's $R$**: $R = \frac{100 \ln N}{1 - V_1/V}$ (sensibile alla percentuale di vocaboli unici usati una sola volta, *hapax legomena*).

---

## 2. Spiegazione Teorica ed Empirica della Correlazione con il Target ($y$)

Calcolando la correlazione di Pearson $r$ su **2.500 testi reali** del dataset ($y=0$ Umano, $y=1$ AI), emergono associazioni marcate che confermano la teoria della letteratura forense:

| Feature Stilometrica | Correlazione $r$ con $y$ | Direzione / Bias | Interpretazione Teorica e Forense |
| :--- | :---: | :---: | :--- |
| **`functional_lexical_ratio`** | **-0.6038** | **Forte Umano** 🧑 | **Il più forte predittore singolo**: gli esseri umani usano un'impalcatura di parole funzionali/connettivi molto più densa. L'AI comprime il contenuto e minimizza le ridondanze funzionali. |
| **`avg_word_len`** | **+0.5189** | **Forte AI** 🤖 | Gli LLM prediligono termini formali, latinismi e vocaboli accademici polisillabici (*revolutionized, sophisticated, demonstrates*). |
| **`ttr` (Type-Token Ratio)** | **+0.3449** | **AI** 🤖 | Gli LLM massimizzano la diversità lessicale locale evitando ripetizioni spontanee. |
| **`brunet_w`** | **-0.3362** | **Umano** 🧑 | Coerente con la formulazione matematica inversa di $W$ rispetto alla diversità lessicale. |
| **`expressive_punct_ratio`** | **+0.3235** | **AI** 🤖 | Gli LLM usano virgolette e trattini con cadenza calibrata nei saggi; gli umani variano molto per genere letterario. |
| **`digit_density`** | **-0.2025** | **Umano** 🧑 | I testi umani (specialmente news e cronaca) includono molte più cifre numeriche concrete, date e percentuali. |
| **`ratio_word_to_sentence_len`** | **+0.1928** | **AI** 🤖 | L'AI impacchetta parole più lunghe all'interno di frasi di lunghezza standardizzata dal prompt. |
| **`punct_density`** | **-0.1533** | **Umano** 🧑 | La scrittura umana complessiva ha una densità di interpunzione superiore di circa un terzo. |

---

## 3. Analisi della Collinearità tra Feature (Inter-Feature Correlation)

L'ispezione della matrice di correlazione individua coppie di feature ad alta collinearità ($|r| \ge 0.60$), elemento cruciale per la successiva fase di **Feature Selection** e regolarizzazione:

```
    ⚠️ Collinear Pair: brunet_w <-> ttr                     (r = -0.9887)
    ⚠️ Collinear Pair: functional_lexical_ratio <-> avg_word_len (r = -0.8193)
    ⚠️ Collinear Pair: syntactic_pause_ratio <-> avg_sentence_len (r = +0.7522)
    ⚠️ Collinear Pair: brunet_w <-> honore_r               (r = -0.7281)
    ⚠️ Collinear Pair: honore_r <-> ttr                     (r = +0.7241)
    ⚠️ Collinear Pair: punct_density <-> quote_density       (r = +0.7141)
    ⚠️ Collinear Pair: ratio_word_to_sentence_len <-> avg_sentence_len (r = -0.6881)
```

### Spiegazione dei Cluster di Collinearità:
1. **Cluster Ricchezza Lessicale (`ttr` $\leftrightarrow$ `brunet_w` $\leftrightarrow$ `honore_r`)**:
   - $r = -0.99$ tra $W$ e TTR: matematicamente $W = N^{V^{-0.172}}$ cattura lo stesso fenomeno. Mantenere entrambi nei modelli lineari causerebbe instabilità nei pesi $\beta$.
2. **Cluster Lunghezza vs Funzionali (`avg_word_len` $\leftrightarrow$ `functional_lexical_ratio`)**:
   - $r = -0.82$: i testi ricchi di parole funzionali contengono lemmi molto brevi (*a, in, of, to*), abbassando automaticamente la lunghezza media della parola.
3. **Cluster Sintassi Periodale (`syntactic_pause_ratio` $\leftrightarrow$ `avg_sentence_len`)**:
   - $r = +0.75$: periodi più lunghi richiedono obbligatoriamente un numero maggiore di virgole per coordinare le proposizioni.

---

## 4. Heatmap di Correlazione Esportata

La mappa termica ad alta risoluzione è disponibile in:
[reports/figures/correlation_heatmap.png](file:///Users/eliagismondi/Desktop/pan-authorship-attribution/reports/figures/correlation_heatmap.png)

Essa illustra visivamente:
- Il gradiente cromatico da blu (forte correlazione negativa, $-1.0$) a rosso (forte correlazione positiva, $+1.0$).
- L'ultima riga/colonna (`target_is_ai`), che evidenzia a colpo d'occhio i predittori stilistici più determinanti per discriminare Umano vs AI.
