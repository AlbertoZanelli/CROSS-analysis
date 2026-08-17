# Handoff — ThalliumStabilization.py, canvas `combined_thallium`

**File di lavoro:** `batchOctopus/ThalliumStabilization.py`
**Impostazioni per canale:** `batchOctopus/chain_settings.csv`
**Dati di test (locali):** `CROSS/MergedRuns/CorrectedAmp/` — canali 26, 27, 57, 58

---

## 0. Come eseguire (IMPORTANTE)

```bash
conda activate pyrootAlbi          # Python 3.12.9 + ROOT 6.34.04
python ThalliumStabilization.py 57
```

Niente `PYTHONPATH=...`, niente `conda run`.

**Perché conta:** il `~/.zshrc` dell'utente fa `source .../root/6.40.02_1/bin/thisroot.sh` (riga 26) e
`setupGarfield.sh` (riga 40), che esportano `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH` e
**`DYLD_LIBRARY_PATH`** verso la ROOT di Homebrew (6.40, compilata per Python 3.14). Risultato:
PyROOT falliva il controllo di versione e — anche dopo aver sistemato `PYTHONPATH` — la ROOT di
conda caricava `libCore` di Homebrew e andava in **segmentation violation**.

Risolto con due hook (NON toccare il `.zshrc`, che serve a Geant4/Garfield):

```
/opt/anaconda3/envs/pyrootAlbi/etc/conda/activate.d/zz_pythonpath_isolate.sh
/opt/anaconda3/envs/pyrootAlbi/etc/conda/deactivate.d/zz_pythonpath_isolate.sh
```

Rimuovono le voci `*/Cellar/root/*` da `PATH`/`LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`/`JUPYTER_PATH`
e azzerano `PYTHONPATH` mentre l'ambiente è attivo; `deactivate` ripristina tutto. Per disfare:
cancellare i due file.

**Nota:** ROOT 6.40 e 6.34 danno risultati DIVERSI (fit gaussiani). Con 6.40 `estimate_thallium_peak`
su ch57 dava σ = 19.2, con 6.34 σ = 26.5 — abbastanza da cambiare quali partizioni si stabilizzano.
Il cluster va verificato con `root-config --version`; `pyrootAlbi` (6.34) riproduce il suo comportamento.

`BASE_DIR` nel file punta al **cluster**. Per test locali fare una copia con il path locale
(non modificare l'originale).

---

## 1. Obiettivo

Nella canvas `chXX_combined_thallium.jpg` confrontare la risoluzione della riga del 208-Tl lungo la
catena di ricostruzione dell'ampiezza. Layout **3 righe × N colonne**:

| riga | contenuto |
|---|---|
| 1 | spettro pieno `[2300·k, 2800·k]` nelle unità native, nessun fit |
| 2 | zoom sul picco + fit, unità native |
| 3 | stesso picco riscalato in energia (picco → 2614.511 keV) + fit |

Colonne (chiavi `chain_defs`): `rough` (calibration_rough) · `heater` (stabilization_all) ·
`corrected` (ampiezza principale) · `stabilized` (ampiezza stabilizzata sul Tl).
Sui canali in `OPTIMUM_FILTER_CHANNELS` restano **solo `rough` e `stabilized`**
(`OPTIMUM_FILTER_CHAIN_KEYS`): non hanno stabilizzazione heater, e la colonna dell'ampiezza
principale ripeterebbe la rough — è la stessa ampiezza dell'optimum filter, a una calibrazione di
distanza. Quei canali non hanno quindi righe `heater`/`corrected` nel CSV dei risultati, e nel plot
di significatività il "prima" diventa `rough` (punti a marker vuoto, vedi `Z_BEFORE_KEYS`).

La canvas è prodotta **due volte**: `chXX_combined_thallium.jpg` (fondo `pol1`) e
`chXX_combined_thallium_flatbkg.jpg` (fondo `pol0`).

### Tabella dei risultati (`SAVE_RES_CSV`)

Ogni pannello fittato finisce in `ThalliumStabilizedAmp/thallium_resolutions.csv`
(formato lungo: una riga per `canale × variabile × riga(native|energy) × fondo(pol1|pol0)`,
con μ, σ, FWHM, risoluzione % e relativi errori, χ²/ndf, P, conteggi). Il file si accumula:
rianalizzare un canale ne **sostituisce** le righe, gli altri restano. Non viene scritto nelle
anteprime della GUI (`show_canvas=True`), solo sul run finale.

I numeri arrivano da `fit_metrics()`, la stessa funzione che riempie i box del fit: CSV e canvas
non possono divergere.

`PlotThalliumResolutions.py` legge quel CSV (default: riga `energy`, fondo `pol0`) e produce due
figure in PDF+PNG, tutto in inglese:

1. risoluzione dei quattro step vs canale, punti dello stesso canale **allineati in verticale**
   (con il rivelatore intero sull'asse x non c'è spazio per sfalsarli);
2. significatività della variazione fra *After second heater stabilization* e
   *After thallium stabilization*: `z = (R_before − R_after)/√(σ_b² + σ_a²)`, positivo se la
   risoluzione migliora, con banda di non-significatività al quantile di Student al 95 % calcolato
   canale per canale (ν efficace alla Welch–Satterthwaite sugli `ndf` dei due fit).
   **I due fit usano gli stessi eventi**, quindi la somma in quadratura sovrastima l'errore della
   differenza: z è conservativo.

Il lettore del CSV segnala i **marcatori di conflitto git** (il file dei risultati viene scritto sia
sul cluster sia in locale, e un merge non risolto lascia dentro *entrambi* i lati) e, se lo stesso
pannello compare più volte, tiene quello con la `date` più recente — così la figura non dipende
dall'ordine delle righe. Gli avvisi vanno su stderr: se compaiono, sistemare il CSV.

---

## 2. Stato attuale — come si determina la finestra di fit

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
nb       = clip( (hi−lo)/(σ_att/bin_div(key)), 15, 200 )   con σ_att = res_exp·μ
```

Fit `gaus(0) + pol1(3)` (o `pol0`), likelihood poissoniana `"Q0 R L"`, con vincoli:

| parametro | limite |
|---|---|
| ampiezza | `[0, 10·max]` |
| media | `centro ± PEAK_MEAN_MAX_SHIFT(2)·σ_att` |
| σ | `[sig_lo, sig_hi]·σ_att` = `[0.3, 1.5]` |
| costante di fondo (**solo pol0**) | `[0, 10·max]`, inizializzata alla media dei bin di bordo |

---

### Flag degli eventi heater

Si chiama `IsHeater` o `heat_IsHeater` a seconda della produzione: `heater_flag_leaf` in
`run_stabilization` prende **quello che il file ha** (`GetLeaf` cerca anche negli alberi amici) e da
lì lo usa il filtro `RDataFrame`. Per aggiungere un terzo nome, la tupla è su quella riga. Se non c'è
nessuno dei due, l'analisi degli heater viene saltata e si usa `CORR_VALID_MIN`, come prima.

## 3. Impostazioni per canale — `chain_settings.csv`

```
channel,win_scale_rough,win_scale_heater,win_scale_corrected,win_scale_stabilized,
        bin_div_rough,bin_div_heater,bin_div_corrected,bin_div_stabilized,
        sig_scale_rough,sig_scale_heater,sig_scale_corrected,sig_scale_stabilized,
        peak_nsigma,sig_lo,sig_hi
```

Aggiungere una colonna è sicuro: all'apertura il file viene **aggiornato in place**, le celle mancanti
riempite con i default del programma (le righe tarate a mano restano tali).

- `USE_CHAIN_CSV = True` → i valori del CSV vincono sui default del programma.
- Il file **si mantiene da solo**: un canale assente viene *aggiunto* coi valori correnti; le righe
  esistenti non vengono **mai** riscritte (così le tarature a mano sono al sicuro). Per resettare un
  canale: cancellarne la riga. Non serve una flag di salvataggio.
- `win_scale_*` allarga la finestra di quel solo pannello (utile per picchi sdoppiati: ch27 rough usa 3).
- `bin_div_*` è la larghezza dei bin, `σ/valore` — più alto = bin più fini.
- `sig_scale_*` moltiplica la **larghezza attesa** di quella sola variabile. La larghezza viene dalle
  partizioni della stabilizzazione, quindi vale per l'ampiezza principale e per la stabilizzata; le
  ampiezze *prima* possono avere una riga molto più larga (calibrazione rough scentrata di qualche %)
  e con la larghezza sbagliata **non sono fittabili affatto**: il fit sbatte sul limite inferiore di σ
  e si aggancia a un singolo bin. Poiché la larghezza attesa fissa binning, vincoli su σ e guinzaglio
  sulla media, è l'unico modo di allargare quel pannello senza toccare gli altri
  (`sig_lo`/`sig_hi` sono per canale, non per variabile).
  Casi reali: **ch60 rough** σ_vera ≈ 22 contro 5.9 attesa → `sig_scale_rough=4.5`, `win_scale_rough=3`,
  `bin_div_rough=2.5` (prima: σ = 1.77 sul limite, risoluzione 0.16 % priva di senso → ora 1.73 ± 0.33 %).
  **ch27 rough** → `sig_scale_rough=2` (prima riga 2 = 0.34 % e riga 3 = 0.90 % sugli stessi eventi,
  segno tipico di un intervallo di σ che ammette sia lo spike sia la riga vera; ora entrambe 1.31 ± 0.24 %).

---

### Partizioni di baseline: popolazioni basse ma non trascurabili

La soglia `PART_GAP_HEIGHT_FRAC` (3 % del bin più alto) è **dipendente dalla scala**: se il picco
principale è stretto e altissimo e l'istogramma è largo — perché esiste un secondo livello di
baseline — il 3 % del massimo supera l'intera popolazione secondaria. Su **ch86** una popolazione a
baseline −6…−3 che contiene il **30 % degli eventi** (1053 dei 1310 eventi di tallio) finiva
inglobata nella partizione esterna: una sola partizione `[-3952, 1.25]`.

Correzione: dopo la ricerca a soglia relativa, l'istogramma viene riletto con una soglia **bassa e
assoluta** (`PART_WARM_MIN_COUNTS = 5`) e un gruppo che non si sovrappone a un blocco già trovato
diventa partizione **solo se il suo integrale** passa lo stesso test `PART_MIN_BLOCK_FRAC` usato per
gli altri blocchi. Ciò che rende trascurabile una popolazione è quanti **eventi** contiene, non
quanto sono alti i suoi bin. Il passo è puramente additivo: non può togliere separazioni esistenti —
verificato, ch26/27/57/58/60 danno partizioni identiche al bit, ch86 passa a 2 partizioni con
confine a −1.71.

Nella canvas delle partizioni i pad 2 e 3 usano ora lo **stesso range robusto** del pad 1
(1°–99° percentile), invece di autoscalare sugli outlier (ch86 aveva un evento a baseline −3952 che
schiacciava tutto su una riga verticale), e vengono disegnati solo i confini **interni**.

## 4. Cosa ha funzionato

- **Ancorare tutto ai picchi di partizione della stabilizzazione.** È l'unico punto della catena dove
  la riga è inequivocabile. Finestra, larghezza attesa e posizione derivano da lì.
- **Esprimere la finestra come FRAZIONE** (`frac`): vale per ogni ampiezza senza conversioni.
- **Finestra simmetrica** (`max` dei due lati): l'unione dei picchi di partizione è asimmetrica ma
  viene applicata attorno al picco locale del pannello; se asimmetrica, il margine bianco resta da un
  lato solo.
- **Vincolo assoluto su σ** `[0.3, 1.5]·σ_attesa`: impedisce alla gaussiana di allargarsi sul fondo.
  Un vincolo *derivato dal seed* non funziona (vedi §5).
- **Ricerca in banda ±4 %** per rough/heater/stabilized: abbastanza larga da recuperare una conversione
  sbagliata del 3 % (ch57 rough), abbastanza stretta da escludere il continuo ai bordi (ch27 heater).
- **Nessuna ricerca sull'ampiezza principale**: lì `mu_exp` è esatto per costruzione.
- **Criterio di significatività** (non conteggi) per accettare il picco locale di una partizione:
  `(N_picco − fondo)/√N_picco ≥ 3`. Con `√fondo` al denominatore un bump di 5 conteggi passava per 5σ.
- **`PART_MIN_CLEAN_EVENTS = 5`**: una partizione con meno eventi puliti eredita la retta della vicina.
  Risolve il crash su ch27 (`ROOT.TGraph(0, …)` → TypeError qui, **segfault** sul cluster).

## 5. Cosa NON ha funzionato (da non ripetere)

- **Cercare il picco senza guida** dentro la finestra di stabilizzazione: su ch27 heater e ch57/ch27
  rough il continuo al bordo è più alto della riga e il finder ci si aggancia (errore fino a 760 unità).
- **Usare solo la conversione** `mu_corr · k/k_main` senza ricerca: su ch57 rough i fattori k sono
  distorti del 3 %, la finestra manca il picco e la seconda passata (media vincolata a ±2σ) non recupera.
- **Ricerca in banda anche sull'ampiezza principale**: su ch58 corrected derivava di +129 (1.9 %) su
  una struttura vicina.
- **Vincolo su σ derivato dal seed** (`[0.6, 2]·seed`): con un seed scentrato la σ si appiccica al
  limite e **tutti i pannelli riportano la stessa risoluzione** — valore privo di significato.
- **Iterare il fit** ri-seedando la larghezza a ogni passata: instabile, oscillava fra σ = 9 e σ = 30.
- **Binning statistico** (~N conteggi/bin): l'utente lo ha provato e preferisce il binning fisso `σ/bin_div`.
- **Vincolare `par3 ≥ 0` su `pol1`**: sbagliato, `par3` è l'intercetta a x = 0 (migliaia di unità fuori
  finestra); non controlla il fondo dentro la finestra. Fatto solo su `pol0`.

---

## 6. Punti aperti

1. **ch27 `calibration_rough` è inutilizzabile.** Sugli stessi 45 eventi di tallio: dispersione q25–q75
   di **1110 unità (47 %)** in rough, contro 32 (0.4 %) in corrected. Non c'è una riga da inquadrare.
   Ipotesi da valutare: rilevare la condizione (confronto della dispersione con quella della corrected)
   e non fittare quel pannello, segnalandolo nel box.
2. **Fondo `pol1` negativo dentro la finestra** in vari pannelli (ch58 heater da +0.42 a −1.21).
   Correzione proposta e non implementata: riparametrizzare la retta attorno al centro,
   `gaus(0) + [3] + [4]*(x−C)`, così `par3` è il fondo al centro e `par3 ≥ 0` è fisico; eventualmente
   vincolare la pendenza con `|par4| ≤ par3/semi-larghezza`.
3. **Righe 2 e 3 divergono** su strutture con spalla (ch27 corrected 0.41 % vs 0.22 %; ch58 stabilized
   0.28 % vs 0.68 %). Una gaussiana singola non le descrive; la divergenza è di per sé un indicatore.
4. **Errori `nan`** su pannelli a bassa statistica (ch58 heater: σ = 15.7 ± nan). Il fit converge ma la
   matrice degli errori no. Valori centrali plausibili, incertezze no.
5. **`CHAIN_PEAK_NSIGMA` è definita due volte** (righe 293–294: `6` e `6.0`). Vince la seconda,
   innocuo, ma va rimossa una delle due.
6. **Canali 25 e 59** (`OPTIMUM_FILTER_CHANNELS`, colonna heater esclusa): la logica è verificata
   forzando ch26 come optimum-filter, ma **non** su dati reali — mancano i loro file in
   `MergedRuns/CorrectedAmp`.

---

## 7. Versioni e backup

| file | cos'è |
|---|---|
| `batchOctopus/ThalliumStabilization.py` | **versione corrente** |
| `batchOctopus/PlotThalliumResolutions.py` | scatter risoluzione vs canale, dal CSV dei risultati |
| `batchOctopus/ThalliumStabilization_.py` | snapshot dell'utente (16 ago); da qui è ripartita la versione attuale |
| `batchOctopus/ThalliumStabilization_backup_oldfit.py` | versione col metodo di fit del programma vecchio |
| `batchOctopus/ThalliumStabilization_old.py` | programma originale (2 pad before/after) |
| `/tmp/before_pos.py`, `/tmp/before_2pass.py` | stadi intermedi di questa sessione (volatili) |

---

## 8. Note di metodo

- L'utente lavora **in parallelo** sullo stesso file: verificare sempre lo stato reale con `grep`/`diff`
  prima di modificare, senza fidarsi di cosa risulta da messaggi precedenti.
- Ogni modifica va verificata su **almeno ch26, 27, 57, 58**: sono i quattro casi che sollecitano
  comportamenti diversi (ch26 statistica buona, ch27 rough rotta, ch57 conversione distorta,
  ch58 struttura vicina al picco).
- Il valore stampato `Counts under the peak` viene dal pannello *stabilized* della riga 3
  (`A·σ·√(2π)/larghezza_bin`) ed è un rapido indicatore di regressione.
