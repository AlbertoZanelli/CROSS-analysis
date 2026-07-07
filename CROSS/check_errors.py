import re
from pathlib import Path

def normalizza_testo(testo):
    """
    Sostituisce qualsiasi sequenza di spazi, tab o ritorni a capo 
    con un singolo spazio per rendere il confronto più robusto.
    """
    return re.sub(r'\s+', ' ', testo).strip()

def controlla_log_errori():
    cartella = "/data/users/azanelli/octopus_work/CROSS/error/"
    
    # Il testo esatto che vogliamo ignorare
    testo_standard = """Currently Loaded Modulefiles:
 1) cmake/3.31.4   3) fftw/3.3.10    5) gsl/2.8        
 2) gcc/15.2.0     4) root/6.32.04   6) python/3.12.3"""
    
    testo_standard_norm = normalizza_testo(testo_standard)
    path = Path(cartella)

    if not path.exists():
        print(f"Errore: La directory {cartella} non esiste.")
        return

    file_trovati = 0
    file_anomali = 0

    print(f"Scansione della cartella: {cartella}\n")

    # Itera su tutti i file nella cartella
    for file_path in path.iterdir():
        if file_path.is_file():
            file_trovati += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    contenuto = f.read()
                
                # Salta i file completamente vuoti (se non ti interessano)
                if not contenuto.strip():
                    continue
                
                # Confronta il contenuto normalizzato
                if normalizza_testo(contenuto) != testo_standard_norm:
                    file_anomali += 1
                    print(f"=== {file_path.name} ===")
                    print(contenuto.strip())
                    print("=" * 60 + "\n")
                    
            except Exception as e:
                print(f"Impossibile leggere il file {file_path.name}: {e}")

    print(f"Controllo terminato: {file_trovati} file analizzati, {file_anomali} contengono errori reali.")

if __name__ == "__main__":
    controlla_log_errori()