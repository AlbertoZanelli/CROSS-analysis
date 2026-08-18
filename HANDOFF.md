# Handoff — stabilizzazione Tl e α, e i plot per la tesi

**Cartella di lavoro:** `batchOctopus/`
**Dati di test locali:** `CROSS/MergedRuns/CorrectedAmp/` — ch **26, 27, 57, 58, 60, 64, 86**

I tre programmi e cosa producono:

| file | cosa fa |
|---|---|
| `ThalliumStabilization.py` | stabilizza sull'ampiezza usando la riga del 208-Tl; canvas `combined_thallium` (3 righe × N colonne) e tabella dei risultati |
| `AlphaStabilization.py` | stabilizza sul doppietto α di 210-Po; in più un **cross-check** che misura la riga del Tl prima e dopo, con la stessa macchina del programma del tallio |
| `PlotThalliumResolutions.py` | legge le tabelle dei risultati e disegna le due figure di tesi (risoluzione vs canale, significatività del miglioramento) |

---

## 0. Come eseguire (IMPORTANTE)

```bash
conda activate pyrootAlbi          # Python 3.12.9 + ROOT 6.34.04
python ThalliumStabilization.py 57
```

Niente `PYTHONPATH=...`, niente `conda run`.

**Perché conta:** il `~/.zshrc` dell'utente fa `source .../root/6.40.02_1/bin/thisroot.sh` e
`setupGarfield.sh`, che esportano `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH` e **`DYLD_LIBRARY_PATH`**
verso la ROOT di Homebrew (6.40, compilata per Python 3.14). Risultato: PyROOT falliva il controllo
di versione e — anche dopo aver sistemato `PYTHONPATH` — la ROOT di conda caricava `libCore` di
Homebrew e andava in **segmentation violation**. Risolto con due hook (NON toccare il `.zshrc`, serve
a Geant4/Garfield):

```
/opt/anaconda3/envs/pyrootAlbi/etc/conda/activate.d/zz_pythonpath_isolate.sh
/opt/anaconda3/envs/pyrootAlbi/etc/conda/deactivate.d/zz_pythonpath_isolate.sh
```

Rimuovono le voci `*/Cellar/root/*` dai path e azzerano `PYTHONPATH` mentre l'ambiente è attivo.

**Nota:** ROOT 6.40 e 6.34 danno risultati DIVERSI sui fit gaussiani (su ch57 σ = 19.2 contro 26.5,
abbastanza da cambiare quali partizioni si stabilizzano). Verificare il cluster con
`root-config --version`; `pyrootAlbi` (6.34) ne riproduce il comportamento.

### Test locali senza sporcare gli output

`BASE_DIR` dei programmi punta al **cluster** (o alla cartella reale): non modificarlo. Per provare in
locale, copia il programma in una sandbox con il `BASE_DIR` sostituito e symlink ai file `.root`:

```bash
S=/tmp/dev; mkdir -p $S/CorrectedAmp
ln -sf <repo>/CROSS/MergedRuns/CorrectedAmp/*ch*_corr.root $S/CorrectedAmp/
cp <repo>/batchOctopus/chain_settings.csv $S/
sed -e "s|^BASE_DIR = .*|BASE_DIR = \"$S/CorrectedAmp\"|" \
    <repo>/batchOctopus/ThalliumStabilization.py > $S/ThalliumStabilization.py
cd $S && python ThalliumStabilization.py 57
```

Tutti gli output (JPEG, ROOT, CSV) finiscono nella sandbox. `CHAIN_CSV_PATH` è relativo alla
posizione dello script, quindi ricordarsi di copiare anche i CSV delle impostazioni.

---

## 1. Obiettivo

Misurare la risoluzione della riga del 208-Tl lungo la catena di ricostruzione dell'ampiezza, canale
per canale, e quantificare quanto la migliorano le due stabilizzazioni (tallio e α). Il prodotto
finale sono due figure per la tesi.

Canvas `chXX_combined_thallium.jpg`, **3 righe × N colonne**:

| riga | contenuto |
|---|---|
| 1 | spettro pieno `[2300·k, 2800·k]` nelle unità native, nessun fit |
| 2 | zoom sul picco + fit, unità native |
| 3 | stesso picco riscalato in energia (picco → 2614.511 keV) + fit |

Colonne nel programma del tallio: `rough` (calibration_rough) · `heater` (stabilization_all) ·
`corrected` (ampiezza principale) · `stabilized` (stabilizzata sul Tl). Sui canali in
`OPTIMUM_FILTER_CHANNELS` restano **solo `rough` e `stabilized`** (`OPTIMUM_FILTER_CHAIN_KEYS`).
Nel programma α le colonne sono due: `corrected` e `alpha`.

Il fondo del fit è **piatto (`pol0`)** ovunque: il fondo lineare è stato rimosso (17 ago), insieme
alla canvas `_flatbkg` parallela. I file `chXX_combined_thallium_flatbkg.jpg` rimasti sui dischi sono
vecchi e si possono cancellare.

---

## 2. Come si determina la finestra di fit (programma del tallio)

Tutto parte dalla stabilizzazione, non da ricerche cieche sullo spettro.

```
chain_peak_interval(part_results, cfg.peak_nsigma)  →  frac, mu_corr, res_exp

  mu_corr = media pesata dei picchi puliti di partizione (unità corrected)
  res_exp = √(sig_in² + spread²)/mu_corr    larghezza intra-partizione ⊕ dispersione fra partizioni
  frac    = semi-ampiezza SIMMETRICA dell'unione dei μᵢ ± peak_nsigma·σᵢ, come FRAZIONE
```

Per ogni variabile:

```
k       = rough_to_units(values, cal_rough, mask_conv)     # = conv_factor per l'ampiezza principale
mu_exp  = mu_corr · k / k_main                             # posizione attesa della riga

se key != "corrected":                                     # rough, heater, stabilized
    ricerca del picco in ±CHAIN_SEARCH_FRAC (4%) attorno a mu_exp
    poi 2 passate: finestra su quel picco → ricentrata sul μ fittato
se key == "corrected":                                     # CHAIN_MAIN_KEY
    nessuna ricerca, nessun ricentraggio: mu_exp È già il picco della stabilizzazione

finestra = μ·(1 ∓ frac·win_scale(key))
nb       = clip( (hi−lo)/(σ_att/bin_div(key)), 15, 200 )   con σ_att = res_exp·sig_scale(key)·μ
```

Fit `gaus(0) + pol0(3)`, likelihood poissoniana `"Q0 R L"`, con vincoli:

| parametro | limite |
|---|---|
| ampiezza | `[0, 10·max]` |
| media | `centro ± PEAK_MEAN_MAX_SHIFT(2)·σ_att` |
| σ | `[sig_lo, sig_hi]·σ_att` = `[0.3, 1.5]` |
| costante di fondo | `[0, 10·max]`, inizializzata alla media dei bin di bordo |

Nel programma **α** la struttura è identica, ma l'ancora è diversa **per forza**: lì le partizioni
sono ancorate alla riga α, quindi il tallio si misura **una volta** sull'ampiezza corretta degli
eventi gamma selezionati (correlazione + taglio LY sul tallio + finestra rough). Quella misura è a
**due passate** — la prima sull'intera finestra, la seconda ristretta a ±3σ: su ch86 la prima dava
σ = 103 (agganciava una struttura larga del continuo), la seconda 35.9, e la risoluzione passa da
0.82 % a 0.33 ± 0.05 %, che coincide con lo 0.311 ± 0.046 % del programma del tallio sulla stessa
ampiezza. Due programmi indipendenti che concordano sono la verifica migliore disponibile.

---

## 3. I file che i programmi si scambiano

Questa è la parte più recente e più importante da capire: i due programmi non sono più indipendenti.

| file | scritto da | letto da | contenuto |
|---|---|---|---|
| `batchOctopus/chain_settings.csv` | Tl | Tl **e α** | impostazioni per canale della catena Tl |
| `batchOctopus/alpha_chain_settings.csv` | α | α | impostazioni per canale del cross-check α (solo la colonna `alpha`) |
| `ThalliumStabilizedAmp/thallium_resolutions.csv` | Tl | plot, **α** | risultati dei fit + **`win_frac`/`res_exp`** = la finestra usata |
| `ThalliumStabilizedAmp/baseline_partitions.csv` | Tl | **α** | intervalli di baseline delle partizioni |
| `AlphaStabilizedAmp/alpha_thallium_resolutions.csv` | α | plot | risoluzione del Tl prima/dopo la stabilizzazione α |
| `AlphaStabilizedAmp/alpha_resolutions.csv` | α | (plot) | risoluzione della **riga α** prima/dopo, con errori |

Cosa condivide il programma α (tutto disattivabile dalle costanti in testa al file):

- `TL_SHARED_CHAIN_CSV` → `win_scale_corrected`, `bin_div_corrected`, `sig_scale_corrected` vengono da
  `chain_settings.csv`: l'ampiezza corretta è **la stessa variabile**, quindi si tara una volta sola;
- `TL_REUSE_THALLIUM_WINDOW` → finestra (`win_frac`, `res_exp`) e posizione del picco vengono dalla
  tabella dei risultati del tallio. `win_frac` sul file porta già dentro il `win_scale` di quel
  programma, quindi viene diviso via e riapplicato per variabile (il pannello α eredita la stessa
  finestra **base**, allargata dal suo fattore);
- `USE_THALLIUM_PARTITIONS` → le partizioni di baseline sono **quelle del tallio**, non ricercate di
  nuovo: le due ricerche giravano su selezioni di eventi diverse e potevano non coincidere, e il
  confronto fra le due stabilizzazioni ha senso solo se gli eventi sono divisi allo stesso modo.

Ognuno di questi ricade sul comportamento locale se il file o la riga del canale mancano, e lo dice
nel print (`measured here` contro `from the thallium results table`). **Attenzione:** funziona solo
se il programma del tallio è stato eseguito su quel canale con la versione attuale — le righe scritte
prima dell'aggiunta di `win_frac`/`res_exp` non hanno quelle colonne.

I numeri di ogni tabella vengono da **una sola funzione** (`fit_metrics()`, e `doublet_metrics()` per
la riga α): i box delle canvas e i CSV non possono divergere.

---

## 4. Impostazioni per canale

`chain_settings.csv`:

```
channel,win_scale_rough,win_scale_heater,win_scale_corrected,win_scale_stabilized,
        bin_div_rough,bin_div_heater,bin_div_corrected,bin_div_stabilized,
        sig_scale_rough,sig_scale_heater,sig_scale_corrected,sig_scale_stabilized,
        peak_nsigma,sig_lo,sig_hi
```

- `USE_CHAIN_CSV = True` → i valori del CSV vincono sui default del programma.
- Il file **si mantiene da solo**: un canale assente viene *aggiunto* coi valori correnti; le righe
  esistenti non vengono **mai** riscritte (le tarature a mano sono al sicuro). Per resettare un
  canale: cancellarne la riga. Aggiungere una colonna è sicuro: il file viene **aggiornato in place**
  con i default nelle celle mancanti.
- `win_scale_*` allarga la finestra di quel solo pannello.
- `bin_div_*` è la larghezza dei bin, `σ/valore` — più alto = bin più fini.
- `sig_scale_*` moltiplica la **larghezza attesa** di quella sola variabile. Serve perché la larghezza
  viene dalle partizioni della stabilizzazione: è giusta per l'ampiezza principale e per la
  stabilizzata, ma le ampiezze *prima* possono avere una riga molto più larga e con la larghezza
  sbagliata **non sono fittabili affatto** (il fit sbatte sul limite inferiore di σ e si aggancia a un
  singolo bin). Casi reali: **ch60 rough** σ_vera ≈ 22 contro 5.9 attesa → `sig_scale_rough=4.5`,
  `win_scale_rough=3`, `bin_div_rough=2.5`; **ch27 rough** → `sig_scale_rough=2`.

`alpha_chain_settings.csv` ha la stessa struttura con le chiavi `corrected` e `alpha`; le colonne
`_corrected` vengono comunque sovrascritte da `chain_settings.csv` (vedi §3).

---

## 5. Cosa ha funzionato

- **Ancorare tutto ai picchi di partizione della stabilizzazione** (o, nel programma α, all'unica
  misura in cui la riga è inequivocabile). Finestra, larghezza attesa e posizione derivano da lì.
- **Esprimere la finestra come FRAZIONE**: vale per ogni ampiezza senza conversioni.
- **Vincolo assoluto su σ** `[0.3, 1.5]·σ_attesa`: impedisce alla gaussiana di allargarsi sul fondo.
- **Ricerca in banda ±4 %** per le variabili diverse dalla principale; **nessuna ricerca** sulla
  principale, dove `mu_exp` è esatto per costruzione.
- **Criterio di significatività** (non conteggi) per accettare il picco locale di una partizione:
  `(N_picco − fondo)/√N_picco ≥ 3`. Con `√fondo` al denominatore un bump di 5 conteggi passava per 5σ.
- **`PART_MIN_CLEAN_EVENTS = 5`**: una partizione con meno eventi puliti eredita la retta della
  vicina. Risolve il crash su ch27 (`ROOT.TGraph(0, …)` → segfault sul cluster).
- **Recupero delle popolazioni di baseline basse ma non trascurabili** (`PART_WARM_MIN_COUNTS`): la
  soglia relativa al bin più alto è dipendente dalla scala, e su ch86 una popolazione col **30 % degli
  eventi** finiva inglobata nella partizione esterna. Ciò che rende trascurabile una popolazione è
  quanti **eventi** contiene, non quanto sono alti i suoi bin. Il passo è additivo: verificato,
  ch26/27/57/58/60 danno partizioni identiche.
- **Escludere gli eventi heater** (programma α): il taglio in correlazione si prende **sopra il
  cluster dell'heater** (`AnalyzeHeaterCorrThreshold`, media + 2σ), come nel programma del tallio. Su
  ch57 il pulser cade a rough 5315, dentro la regione di ricerca α, dov'era l'**85 % degli eventi**:
  la stabilizzazione si ancorava al pulser (hint "amplitude 10000.3, sigma 5.5" — 10000 esatto è il
  valore nominale dell'heater).
- **Vincolo fisico sulla separazione del doppietto α** (`DOUBLET_SEP_REL` ≈ 1.91 %, banda a fattore
  2): senza, la finestra si costruisce su due strutture scorrelate.
- **Bin più larghi nei fit del doppietto** (`DOUBLET_BIN_DIV = 2.0`): su ch57 la partizione 0 ha ~360
  eventi e con bin da σ/4 il fit congiunto descriveva due volte la riga bassa.
- **`ld_usable()`**: un LD con meno di 20 eventi a light yield **non nullo** non è selezionabile per
  il taglio LY. Su ch60 `LD2_LY` esiste ma vale zero su tutti i 254793 eventi, e quel picco
  all'origine vinceva il confronto "meglio risolto".

## 6. Cosa NON ha funzionato (da non ripetere)

- **Cercare il picco senza guida** dentro la finestra di stabilizzazione: il continuo al bordo è più
  alto della riga e il finder ci si aggancia (errore fino a 760 unità).
- **Usare solo la conversione** `mu_corr · k/k_main` senza ricerca: su ch57 rough i fattori k sono
  distorti del 3 %, la finestra manca il picco.
- **Ricerca in banda anche sull'ampiezza principale**: su ch58 corrected derivava di +129 (1.9 %).
- **Vincolo su σ derivato dal seed**: con un seed scentrato la σ si appiccica al limite e **tutti i
  pannelli riportano la stessa risoluzione**.
- **Iterare il fit** ri-seedando la larghezza a ogni passata: oscillava fra σ = 9 e σ = 30.
- **Binning statistico** (~N conteggi/bin): l'utente preferisce il binning fisso `σ/bin_div`.
- **Fondo lineare (`pol1`)**: rimosso. `par3` è l'intercetta a x = 0, fuori finestra, non vincolabile
  in modo fisico, e la retta scendeva sotto zero dentro la finestra (ch58 heater da +0.42 a −1.21).
- **Alzare `sig_hi` per tutto il canale** invece di `sig_scale` sulla singola variabile: misurato su
  ch60 — sistema il rough (1.68 %) ma su heater e corrected la gaussiana si allarga a mangiarsi il
  fondo (0.82 → 1.48 % e 0.82 → 1.53 %) e l'errore dello stabilized triplica.
- **Ordinare le coppie di picchi per prominenza combinata** (doppietto α): due bump del continuo
  battono la riga vera per mezza unità. Va ancorata al picco più prominente.

---

## 7. Punti aperti / prossimi passi

**Da fare per primo — rigenerare le tabelle.**

1. `CROSS/MergedRuns/ThalliumStabilizedAmp/thallium_resolutions.csv` contiene **marcatori di
   conflitto git** (`<<<<<<< HEAD` riga 2, `=======` riga 171), committati così nel merge `4450a1a`:
   il file ha dentro **entrambi i lati** del merge e ~60 pannelli duplicati. Il programma di plot lo
   segnala su stderr e usa la riga con la `date` più recente, ma va ripulito.
   ```bash
   grep -n '^<<<<<<<\|^=======\|^>>>>>>>' CROSS/MergedRuns/ThalliumStabilizedAmp/thallium_resolutions.csv
   ```
2. Le righe esistenti sono state scritte **prima** delle colonne `win_frac`/`res_exp` e prima del
   `baseline_partitions.csv`: finché non si rilancia `ThalliumStabilization.py`, il programma α non
   può condividere né finestra né partizioni. **Rilanciare il tallio su tutti i canali, poi le α.**
3. `CHANNELS_TO_PROCESS` dei due programmi sono **disgiunti** (Tl: 19-24, 49-54, 61-66, 85-90; α:
   25-30, 55-60). Perché la condivisione e il plot combinato funzionino, i canali α devono essere
   analizzati **anche** dal programma del tallio.

**Difetti noti, non risolti.**

4. **ch27 `calibration_rough` è quasi inutilizzabile**: dispersione q25–q75 di 1110 unità (47 %) in
   rough contro 32 (0.4 %) in corrected. Ora è fittabile (`sig_scale_rough=2` → 1.31 ± 0.24 %) ma
   resta una riga larghissima.
5. **Calibrazione rough sospetta nelle ultime run del merge.** Il rapporto
   `corrected_amplitude/calibration_rough` è stabile (3.43, IQR 0.008) per le prime run e diventa
   incoerente evento per evento dalla run 123 in poi; la run 120 dà 0.0918 con IQR 0.0004 (costante,
   ma un fattore 37 diversa). Siccome la colonna `rough` è "Optimum filter amplitude" nel plot di
   tesi, va controllato prima di pubblicare quei numeri.
6. **Errori `nan`** su pannelli a bassa statistica: il fit converge ma la matrice degli errori no.
7. **Righe 2 e 3 divergono** su strutture con spalla: una gaussiana singola non le descrive; la
   divergenza è di per sé un indicatore.
8. **`CHAIN_PEAK_NSIGMA` è definita due volte** in `ThalliumStabilization.py` (righe 318–319: `6` e
   `6.0`). Vince la seconda, innocuo, ma va rimossa una delle due.
9. **ch30** riporta 0.024 % (≈ 0.6 keV a 2615 keV): impossibile, è un fit degenere. Candidato per
   `EXCLUDE_CHANNELS` in `PlotThalliumResolutions.py` o per una taratura in `chain_settings.csv`.
10. **Canali 25 e 59** (`OPTIMUM_FILTER_CHANNELS`): la logica delle due colonne è verificata forzando
    ch26 come optimum-filter, ma **non** su dati reali. Le loro righe nella tabella sono vecchie e
    contengono ancora `corrected`.
11. Nel cross-check α il **taglio LY se lo calcola ognuno** dei due programmi: il "prima" delle α e il
    pannello corrected del tallio restano due misure distinte (ch26: 0.346 ± 0.063 contro
    0.368 ± 0.067, dentro l'errore). Per farle coincidere esattamente andrebbe condiviso anche quello.
12. Il pad 1 della canvas delle partizioni schiaccia le popolazioni secondarie (300 contro 30000
    conteggi): una scala log-y le renderebbe visibili, ma cambia l'aspetto per tutti i canali.

---

## 8. Il programma di plot

`PlotThalliumResolutions.py` legge **due** tabelle: quella del tallio (`--csv`) e, se c'è, quella
delle α (`../AlphaStabilizedAmp/alpha_thallium_resolutions.csv`, oppure `--alpha-csv`; `--no-alpha`
la ignora). Produce due PNG a 300 dpi (niente PDF), tutto in inglese:

1. **risoluzione** dei cinque step vs canale, punti dello stesso canale **allineati in verticale**
   (con il rivelatore intero sull'asse x non c'è spazio per sfalsarli); `--ymax` limita l'asse e i
   punti fuori scala restano marcati con un caret sul bordo;
2. **significatività** `z = (R_before − R_after)/√(σ_b² + σ_a²)`, positivo se la risoluzione migliora,
   con banda di non-significatività al quantile di Student al 95 % calcolato canale per canale (ν
   efficace alla Welch–Satterthwaite). **Due confronti per canale** (`COMPARISONS`): cerchio =
   stabilizzazione tallio, triangolo = α, colore per il segno, sfalsati di ±0.17.
   **I due fit usano gli stessi eventi**, quindi la somma in quadratura sovrastima l'errore della
   differenza: z è conservativo.

`EXCLUDE_CHANNELS` in testa al file esclude canali da entrambe le figure (vince su tutto, e stampa
quali ha tolto).

---

## 9. Versioni, backup, note di metodo

| file | cos'è |
|---|---|
| `batchOctopus/ThalliumStabilization.py` | **versione corrente** |
| `batchOctopus/AlphaStabilization.py` | **versione corrente** |
| `batchOctopus/PlotThalliumResolutions.py` | **versione corrente** |
| `batchOctopus/ThalliumStabilization_.py` | snapshot dell'utente (16 ago) |
| `batchOctopus/ThalliumStabilization_backup_oldfit.py` | versione col metodo di fit del programma vecchio |
| `batchOctopus/ThalliumStabilization_old.py` | programma originale (2 pad before/after) |

- L'utente lavora **in parallelo** sugli stessi file: verificare sempre lo stato reale con
  `grep`/`diff` prima di modificare, senza fidarsi di cosa risulta da messaggi precedenti.
- Ogni modifica va verificata su **almeno ch26, 27, 57, 58** (più 60, 64, 86 ora disponibili): sono i
  casi che sollecitano comportamenti diversi — ch26 statistica buona, ch27 rough rotta, ch57
  conversione distorta e heater dentro la regione α, ch58 struttura vicina al picco, ch60 LD2 assente,
  ch86 secondo livello di baseline.
- `Counts under the peak` (stampato a fine canale) è un rapido indicatore di regressione.
- I file `.root` di test sono grandi (100–200 MB) e un canale impiega 1–3 minuti: per confronti
  prima/dopo conviene catturare l'output di riferimento **prima** di modificare.
