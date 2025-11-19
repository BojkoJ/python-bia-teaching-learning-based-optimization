# BIA – Cvičení 10 (Biologicky inspirované algoritmy)

Tento repozitář obsahuje řešení 10. cvičení z předmětu Biologicky inspirované algoritmy.

Implementace zahrnuje **5 optimalizačních algoritmů** a jejich porovnání na 9 testovacích funkcích v 30-rozměrném prostoru.

## Obsah

- **`main.py`** – Implementace všech algoritmů a testovacích funkcí:

  - **Algoritmy**: Differential Evolution (DE), Particle Swarm Optimization (PSO), SOMA AllToOne, Firefly Algorithm (FA), Teaching-Learning Based Optimization (TLBO)
  - **Testovací funkce (minimizační)**: Sphere, Schwefel, Rosenbrock, Rastrigin, Griewank, Levy, Michalewicz, Zakharov, Ackley
  - **Pomocné funkce**: `get_default_bounds()`, `generate_grid()`, `evaluate_surface_2d()`, `plot_surface_2d()`

- **`tests/`** – Testovací skripty:

  - `tlbo_test.py` – Vizualizace Teaching-Learning Based Optimization na různých funkcích

- **`run_experiments.py`** – Experimentální skript:
  - Spouští všech 5 algoritmů na všech 9 funkcích
  - Provádí 30 opakovaných experimentů pro každou kombinaci
  - Generuje Excel soubor `tlbo_comparison_results.xlsx` s detailními výsledky
  - Počítá Mean a Std dev pro každý algoritmus

## Algoritmus

### **Teaching-Learning Based Optimization (TLBO)**

- Inspirován procesem výuky ve třídě
- Dvě fáze v každé iteraci:
  1. **Teacher Phase**: Studenti se učí od učitele (nejlepšího řešení)
  2. **Learner Phase**: Studenti se učí od sebe navzájem
- **Klíčová výhoda**: Žádné parametry k ladění (kromě NP a max_OFE)
- Vzorce:
  - Teacher: `X_new = X_old + r * (X_teacher - TF * X_mean)`
  - Learner: `X_new = X_i + r * (X_j - X_i)` nebo `X_new = X_i + r * (X_i - X_j)`

## Jak spustit

### Vizualizace jednotlivých algoritmů (2D funkce)

```powershell
# Firefly Algorithm
python .\tests\firefly_test.py

# Teaching-Learning Based Optimization
python .\tests\tlbo_test.py
```

### Spuštění experimentů (30D, 30 opakování, 5 algoritmů, 9 funkcí)

```powershell
# Vyžaduje: pip install openpyxl numpy matplotlib
python .\run_experiments.py

# Vygeneruje: tlbo_comparison_results.xlsx
```

## Experimentální nastavení

```python
D = 30              # počet dimenzí
NP = 30             # velikost populace
MAX_OFE = 3000      # max počet vyhodnocení cílové funkce
NUM_EXPERIMENTS = 30  # opakovaní pro každou kombinaci
```

**Výstup**: Excel soubor s 9 sheety (jedna tabulka pro každou testovací funkci)

- Řádky: 30 experimentů + Mean + Std dev
- Sloupce: DE, PSO, SOMA, FA, TLBO
- Formát: Normální desetinná čísla (12 decimálních míst)

## Výsledky

TLBO dosáhl **nejlepších výsledků na 6 z 9 testovacích funkcí**:

| Funkce         | TLBO umístění | Poznámka                   |
| -------------- | ------------- | -------------------------- |
| **Sphere**     | 1.            | Prakticky globální optimum |
| Schwefel       | 3.            | Komplikovaná multimodální  |
| **Rosenbrock** | 1.            | Údolí - TLBO dominuje      |
| **Rastrigin**  | 1.            | Mnoho lokálních minim      |
| **Griewank**   | 1.            | Prakticky globální optimum |
| **Levy**       | 1.            | Rychlá konvergence         |
| Michalewicz    | 2.            | Těsně za DE                |
| Zakharov       | 2.            | FA překvapil               |
| **Ackley**     | 1.            | Prakticky globální optimum |
