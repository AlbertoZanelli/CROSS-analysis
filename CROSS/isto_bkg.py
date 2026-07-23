import matplotlib.pyplot as plt
import numpy as np

# Categorie sull'asse Y (ordinate dal basso verso l'alto come nell'immagine)
categories = [
    'Total',
    'Pileup',
    'Muons',
    'Neutrons',
    'Cryostat and Shields',
    'Close Components',
    'Crystals'
]

# Valori corrispondenti estratti (in unità ckky)
values = [
    1.00e-4,
    0.50e-4,
    0.01e-4,
    0.02e-4,
    0.10e-4,
    0.25e-4,
    0.12e-4
]

# Definizione dei colori: #FF4500 (OrangeRed molto brillante) per il Pileup, grigio per il resto
colors = ['#FF4500' if cat == 'Pileup' else '#A9A9A9' for cat in categories]

# Creazione della figura più stretta in orizzontale (6x5 invece di 9x5)
fig, ax = plt.subplots(figsize=(6, 5))

# Creazione del grafico a barre orizzontali assegnando la lista 'colors'
bars = ax.barh(categories, values, color=colors, edgecolor='black', linewidth=0.5)

# Impostazione della scala logaritmica per l'asse X
ax.set_xscale('log')

# Limiti dell'asse X per centrare bene le barre
ax.set_xlim(8e-7, 2.5e-4)

# Configurazione della griglia 
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
ax.set_axisbelow(True)

# Etichetta dell'asse X
ax.set_xlabel('BI [ckky]', fontsize=11)

# Titolo del grafico
ax.set_title('CUPID Background Budget - Total BI = 1·10⁻⁴ckky', fontsize=12, pad=15)

# Ciclo per aggiungere i valori testuali alla fine di ogni barra
for bar in bars:
    width = bar.get_width()
    
    # Calcolo del moltiplicatore per il formato testuale (.xx * 10^-4)
    label_val = width * 10000 
    
    # Creazione della stringa di testo
    label_text = f'{label_val:.2f}·10⁻⁴'
    
    # Posizionamento del testo (spostato leggermente a destra della barra, offset aumentato a 1.1 per via della larghezza ridotta)
    ax.text(width * 1.1, bar.get_y() + bar.get_height() / 2, label_text,
            va='center', ha='left', fontsize=10)

# Ottimizzazione degli spazi
plt.tight_layout()

# Mostra il grafico
plt.show()