import sys
import os
import pandas as pd

# ============================================================
#   CONFIGURAR PYTHONPATH PARA QUE RECONOZCA src/
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT, "src")

# Añadir src al path de Python (solo una vez)
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ============================================================
#   IMPORTAR ESCENARIOS
# ============================================================
from src.simulations.scenario1_ml_simulation import run_ml_simulation
from src.simulations.scenario2_cellular_automata import run_forest_cover_simulation
from src.simulations.scenario2_cellular_automata_realistic import run_forest_cover_simulation2



# ============================================================
#   CARGAR DATASET PRINCIPAL
# ============================================================
def load_dataset():
    """
    Carga el dataset principal que será usado por el Escenario 1.
    """
    data_path = os.path.join(SRC_PATH, "data", "processed", "train_transformed.csv")

    print("Cargando dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

    return df


# ============================================================
#   MAIN
# ============================================================
def main():

    print("\n==============================")
    print("   Forest Cover Simulation")
    print("==============================\n")

    # -----------------------------------
    # ESCENARIO 1 — Machine Learning
    # -----------------------------------
    print("[Escenario 1] Ejecutando simulación ML...")

    df = load_dataset()
    ml_results = run_ml_simulation(df)

    print("Resultados ML:", ml_results)

    # -----------------------------------
    # ESCENARIO 2 — Simulación del bosque
    # -----------------------------------
    print("\n[Escenario 2] Simulando mapa de bosque...")

    forest_map = run_forest_cover_simulation(
        size=120,    # Cambia el tamaño del bosque
        steps=25,    # Número de iteraciones del cellular automata
        show_plot=True
    )
  

    print("\nSimulación completada.")
    print("Dimensiones del mapa final:", forest_map.shape)


# ============================================================
#   EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    main()
