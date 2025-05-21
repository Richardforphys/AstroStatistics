import subprocess
import time

# Interprete Python del tuo ambiente virtuale
python_exe = r"C:\Users\ricca\Documents\Unimib-Code\Astrostatistics\Notebooks\venv\Scripts\python.exe"

# Script di simulazione
script_path = r"C:\Users\ricca\Documents\Unimib-Code\Astrostatistics\Notebooks\venv\Rivelatori\MCPU.py"

# Lista dei valori di d da simulare
d_values = [1, 10, 100]

# Loop
for d in d_values:
    print(f"\nEseguo simulazione con d = {d} cm...")

    start = time.time()  # Avvia timer

    try:
        subprocess.run([python_exe, script_path, str(d)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore nella simulazione con d = {d}: {e}")
    else:
        elapsed = time.time() - start
        print(f"→ Simulazione completata in {elapsed:.2f} secondi")
