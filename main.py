import numpy as np
import math
import random
from typing import Callable, List, Tuple, Optional

# Výchozí hranice (výřezy) pro 2D vizualizaci (X,Y).
DEFAULT_BOUNDS_2D = {
    "sphere": (-5.0, 5.0),          
    "schwefel": (-500.0, 500.0),    
    "rosenbrock": (-5.0, 10.0),   
    "rastrigin": (-5.12, 5.12),
    "griewank": (-6.0, 6.0),
    "levy": (-3.0, 3.0),
    "michalewicz": (0.0, float(math.pi)), 
    "zakharov": (-5.0, 5.0),
    "ackley": (-32.768, 32.768),
}

def get_default_bounds(func_name: str, dim: int = 2) -> List[Tuple[float, float]]:
        """
        Vrátí seznam hranic pro každou dimenzi.

        Parametry:
            func_name : název funkce
            dim       : kolik dimenzí (např. 2 > chceme dvě dvojice hranic).

        Návrat:
            list dvojic (low, high). Každá dvojice jsou float hodnoty dolní a horní meze.
        """
        # Převedeme název na malá písmena
        name_lower = func_name.lower()

        # Zkusíme v tabulce DEFAULT_BOUNDS_2D najít položku podle klíče name_lower, pokud není dáme default
        bounds_pair = DEFAULT_BOUNDS_2D.get(name_lower, (-5.0, 5.0))

        # Rozbalíme dvojici (low, high) do dvou proměnných pro přehlednost.
        # Příklad: bounds_pair = (-5.0, 5.0) -> low = -5.0, high = 5.0
        low = bounds_pair[0]
        high = bounds_pair[1]

        result: List[Tuple[float, float]] = []

        for i in range(dim):
                result.append((low, high))

        return result


def generate_grid(bounds_2d: List[Tuple[float, float]], num_points: int = 100):
    """
    Vytvoří mřížku (X, Y) v zadaných 2D hranicích. Každá z os má `num_points` vzorků.
    """
    (x_min, x_max), (y_min, y_max) = bounds_2d
    
    # np.linspace vytvoří jednorozměrné pole s rovnoměrně rozloženými body
    # mezi zadanými hranicemi. Např. linspace(0, 10, 5) → [0, 2.5, 5, 7.5, 10]
    x = np.linspace(x_min, x_max, num_points) # Vytvoří num_points bodů na x-ose
    y = np.linspace(y_min, y_max, num_points) # Vytvoří num_points bodů na y-ose
    
    # np.meshgrid vezme dva 1D vektory a vytvoří z nich 2D mřížku souřadnic
    # X obsahuje x-souřadnice pro každý bod mřížky
    # Y obsahuje y-souřadnice pro každý bod mřížky
    # Výsledek: každý bod [X[i,j], Y[i,j]] reprezentuje jeden bod v 2D mřížce
    X, Y = np.meshgrid(x, y)
    return X, Y

def evaluate_surface_2d(objective: Callable[[List[float]], float],
                            bounds_2d: List[Tuple[float, float]],
                            num_points: int = 100):
        """
        Vyhodnotí funkci na 2D mřížce: funkce dostane 2 parametry (x, y) a vrátí třetí (z).
        Používáme numpy.vectorize pro čitelnost.
        """
        # Vytvoříme 2D mřížku bodů (X, Y) v zadaných hranicích
        X, Y = generate_grid(bounds_2d, num_points)
        
        # np.vectorize umožňuje aplikovat obyčejnou funkci na numpy pole element po elementu
        # lambda x, y: objective([float(x), float(y)]) převede každou dvojici (x,y) z mřížky
        # na seznam [x, y] a předá ho cílové funkci
        f_vec = np.vectorize(lambda x, y: objective([float(x), float(y)]))
        
        # Aplikujeme vektorizovanou funkci na celé pole X a Y najednou
        # Výsledek Z obsahuje hodnotu funkce pro každý bod mřížky
        # Z[i,j] = objective([X[i,j], Y[i,j]])
        Z = f_vec(X, Y)
        
        return X, Y, Z


def plot_surface_2d(objective: Callable[[List[float]], float],
                    bounds_2d: Optional[List[Tuple[float, float]]] = None,
                    num_points: int = 100,
                    title: Optional[str] = None):
    """
    Jednoduchá 3D vizualizace povrchu funkce (2 vstupy -> 3D graf X,Y,Z).
    Importujeme matplotlib až zde, aby to nebylo povinné při běžných výpočtech.
    """
    if bounds_2d is None:                               # Pokud uživatel nezadal vlastní hranice tak default
        bounds_2d = get_default_bounds("sphere", 2)

    # Vyhodnotíme funkci na pravidelné mřížce bodů (dostaneme X, Y souřadnice a Z hodnoty funkce)
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points)

    import matplotlib.pyplot as plt                      # Import knihovny pro vykreslování

    fig = plt.figure(figsize=(8, 6))                     # Vytvoří novou figuru s danou velikostí
    ax = fig.add_subplot(111, projection='3d')           # Přidá 3D subplot

    # Samotné vykreslení 3D povrchu (surface). Jednotlivé parametry určují barvy a mřížku.
    ax.plot_surface(
        X,                                             # 2D pole x souřadnic
        Y,                                             # 2D pole y souřadnic
        Z,                                             # 2D pole hodnot funkce f(x,y)
        cmap='jet',                                    # Barevná mapa
        edgecolor='k',                                 # Černé hrany každého malého polygonu
        linewidth=0.2,                                 # Tloušťka čar hran
        antialiased=True,                              # Vyhlazení hran pro hezčí vzhled
        rstride=1,                                     # Vykreslit každou řádku mřížky
        cstride=1,                                     # Vykreslit každý sloupec mřížky
    )
    
    ax.set_xlabel('x1')                                 # Popisek osy X
    ax.set_ylabel('x2')                                 # Popisek osy Y
    ax.set_zlabel('f(x)')                               # Popisek osy Z
    
    if title:                                           # Pokud je předán titulek nastavíme ho nad graf
        ax.set_title(title)                             

    plt.tight_layout()                                  # Úprava rozložení (aby se popisky nepřekrývaly)
    plt.show()                                          # Zobrazení okna s grafem

# Jednotlivé testovací funkce:

def sphere(params):
    """
    Definice:
        f(x) = sum(x_i^2) pro i = 1..n
    
    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-5.12, 5.12] pro všechna i = 1, ..., d
    - Globální minimum: x* = (0, 0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Sphere pro zadané params
    """
    total = 0.0
    for value in params:
        total += value * value
    return float(total)

def schwefel(params):
    """
    Definice:
        f(x) = 418.9829 * n - sum_{i=1..n}  x_i * sin(sqrt(|x_i|))]

    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-500, 500]
    - Globální minimum: x_i ≈ 420.968746... pro všechny i, hodnota f(x*) ≈ 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Schwefel pro zadané params
    """
    n = 0
    suma = 0.0
    for value in params:
        n += 1
        # Použijeme absolutní hodnotu uvnitř odmocniny podle definice.
        term = value * math.sin(math.sqrt(abs(value)))
        suma += term

    konst = 418.9829
    result = konst * n - suma
    return float(result)


def rosenbrock(params):
    """
    Definice:
        f(x) = sum_{i=1..n-1} [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    - n je rozměr vektoru x (pro n < 2 je součet prázdný -> 0)
    - Obvyklá doména: x_i v intervalu přibližně [-2.5, 2.5] (často uváděno), někdy [-5, 10]
    - Globální minimum: x* = (1, 1, ..., 1) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rosenbrock pro zadané params
    """
    total = 0.0
    # Součet jde od i=0 do i=n-2 (tj. pracujeme vždy s dvojicí x_i a x_{i+1})
    n = len(params)
    for i in range(0, n - 1):
        xi = params[i]
        x_next = params[i + 1]
        # 100 * (x_{i+1} - x_i^2)^2
        first = 100.0 * (x_next - (xi * xi)) ** 2
        # (1 - x_i)^2
        second = (1.0 - xi) ** 2
        total += first + second

    return float(total)

def rastrigin(params):
    """
    Definice pro n-rozměrný vektor x:
        f(x) = 10 * n + sum_{i=1..n} [ x_i^2 - 10 * cos(2π x_i) ]

    - Obvyklá doména: x_i v intervalu [-5.12, 5.12]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rastrigin pro zadané params
    """
    n = 0
    total = 0.0
    for value in params:
        n += 1
        total += (value * value) - 10.0 * math.cos(2.0 * math.pi * value)

    result = 10.0 * n + total
    return float(result)


def griewank(params):
    r"""
    Definice pro n-rozměrný vektor x:
        f(x) = 1 + \sum_{i=1..n} (x_i^2 / 4000) - \prod_{i=1..n} cos\left(\frac{x_i}{\sqrt{i}}\right)

    - Obvyklá doména: x_i v intervalu [-600, 600]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Griewank pro zadané `params`.
    """
    sum_term = 0.0
    prod_term = 1.0

    # i číslujeme od 1 kvůli definici s odmocninou i
    i = 1
    for value in params:
        sum_term += (value * value) / 4000.0
        prod_term *= math.cos(value / math.sqrt(i))
        i += 1

    result = 1.0 + sum_term - prod_term
    return float(result)


def levy(params):
    r"""
    Standardní (vícerozměrná) Levy funkce.

    Definice:
        Nejprve se provede transformace
            w_i = 1 + (x_i - 1) / 4

        f(x) = sin^2(π w_1)
               + Σ_{i=1..n-1} (w_i - 1)^2 * [ 1 + 10 * sin^2(π w_i + 1) ]
               + (w_n - 1)^2 * [ 1 + sin^2(2π w_n) ]

    Vlastnosti:
        - Doména obvykle x_i ∈ [-10, 10]
        - Globální minimum: x* = (1, ..., 1) → f(x*) = 0

    Parametry
    ---------
    params : list[float]
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota Levy funkce pro zadané `params`.
    """
    n = len(params)
    if n == 0:
        return 0.0

    # Transformace w_i
    w = []
    for x in params:
        w.append(1.0 + (x - 1.0) / 4.0)

    # První člen
    total = math.sin(math.pi * w[0]) ** 2

    # Prostřední součet (i = 1..n-1 => indexy 0..n-2)
    for i in range(0, n - 1):
        wi = w[i]
        total += (wi - 1.0) * (wi - 1.0) * (1.0 + 10.0 * (math.sin(math.pi * wi + 1.0) ** 2))

    # Poslední člen
    wn = w[-1]
    total += (wn - 1.0) * (wn - 1.0) * (1.0 + (math.sin(2.0 * math.pi * wn) ** 2))

    return float(total)


def michalewicz(params):
    r"""
    Definice pro n-rozměrný vektor x (obvykle s parametrem m = 10):
        f(x) = - \sum_{i=1..n} [ sin(x_i) * ( sin( i * x_i^2 / π ) )^{2m} ]

    - Obvyklá doména: x_i v intervalu [0, π]
    - Typické nastavení: m = 10 (čím větší m, tím více lokálních minim)
    - Globální minimum pro n=2, m=10 je přibližně f(x*) ≈ -1.8013 v bodě x* ≈ (2.20, 1.57)

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Michalewicz pro zadané `params` (při m = 10).
    """
    m = 10
    total = 0.0
    i = 1
    for x in params:
        s1 = math.sin(x)
        s2 = math.sin(i * (x * x) / math.pi)
        term = s1 * (s2 ** (2 * m))
        total += term
        i += 1
    return float(-total)


def zakharov(params):
    r"""
    Funkce Zakharov (minimalizační úloha).

    Definice pro n-rozměrný vektor x:
        f(x) = \sum_{i=1..n} x_i^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^4

    - Obvyklá doména: x_i v intervalu [-5, 10] (různé zdroje uvádí mírně odlišně)
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Zakharov pro zadané `params`.
    """
    sum_sq = 0.0
    sum_lin = 0.0
    i = 1
    for x in params:
        sum_sq += x * x
        sum_lin += 0.5 * i * x
        i += 1

    result = sum_sq + (sum_lin ** 2) + (sum_lin ** 4)
    return float(result)


def ackley(params):
    """
    Funkce Ackley (minimalizační úloha).

    Pro n-rozměrný vektor x a konstanty a=20, b=0.2, c=2π:
        f(x) = -a * exp(-b * sqrt( (1/n) * sum(x_i^2) ))
               - exp( (1/n) * sum( cos(c * x_i) ) )
               + a + e

    - Obvyklá doména: x_i v intervalu [-32.768, 32.768]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Ackley pro zadané `params`.
    """
    n = 0
    sum_sq = 0.0
    sum_cos = 0.0
    for x in params:
        n += 1
        sum_sq += x * x
        sum_cos += math.cos(2.0 * math.pi * x)

    if n == 0:
        return 0.0

    a = 20.0
    b = 0.2
    # c = 2*pi je použito přímo v cyklu výše

    term1 = -a * math.exp(-b * math.sqrt((1.0 / n) * sum_sq))
    term2 = -math.exp((1.0 / n) * sum_cos)
    result = term1 + term2 + a + math.e
    return float(result)

def firefly_algorithm(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 20,  # počet světlušek v populaci
    beta_0: float = 1.0,  # β₀ - maximální atraktivita při vzdálenosti r=0
    alpha: float = 0.3,  # α - parametr náhodného pohybu (randomizace)
    max_generations: int = 50,  # maximální počet generací
    seed: Optional[int] = None,  # náhodný seed pro reprodukovatelnost
    visualize: bool = False,  # vizualizace (jen 2D)
    num_points: int = 200,  # hustota mřížky pro heatmapu
) -> Tuple[List[float], float]:
    """
    Firefly Algorithm (FA) - algoritmus inspirovaný chováním světlušek.
    
    Princip algoritmu:
      Světlušky jsou přitahovány k jasnějším světluškám. Intenzita světla závisí na hodnotě
      cílové funkce (fitness) - čím lepší hodnota (menší pro minimalizaci), tím jasnější světluška.
      
    Algoritmus:
      1. Inicializujeme populaci světlušek s náhodnými pozicemi v daných mezích
      2. Pro každou generaci:
         a) Pro každou světlušku i:
            - Pro každou světlušku j:
              * Vyhodnotíme intenzitu světla I_i a I_j (určeno fitness hodnotou)
              * Pokud je j jasnější než i (I_j > I_i, tj. f(j) < f(i) pro minimalizaci):
                - Vypočítáme vzdálenost r_ij mezi světluškami i a j (Euklidovská vzdálenost)
                - Vypočítáme atraktivitu β = β₀/(1+r) (zjednodušený vzorec místo β₀*e^(-γ*r²))
                - Posuneme světlušku i směrem k j s přidáním náhodného pohybu
         b) Vyhodnotíme nové pozice a aktualizujeme intenzitu světla
         c) Najdeme nejlepší světlušku (s nejnižší fitness hodnotou)
      3. Vrátíme nejlepší nalezené řešení
    
    Klíčové parametry:
      - β₀ (beta_0): Maximální atraktivita při r=0. Určuje sílu přitažlivosti. Výchozí: 1.0
      - α (alpha): Váha náhodného pohybu (exploration vs exploitation). Rozsah [0,1]. Výchozí: 0.3
        * Vyšší α = více exploration (průzkum), nižší α = více exploitation (využití)
    
    Rovnice pohybu světlušky:
      x_i(t+1) = x_i(t) + β * (x_j(t) - x_i(t)) + α * ε
      
      kde:
        x_i       = pozice světlušky i
        x_j       = pozice světlušky j (jasnější světluška)
        β         = atraktivita závislá na vzdálenosti: β = β₀/(1+r)
        r         = vzdálenost mezi i a j (Euklidovská)
        α         = parametr náhodného pohybu
        ε         = náhodný vektor z normálního rozdělení N(0,1)
    
    Vizualizace (jen 2D):
      - Heatmapa cílové funkce na pozadí
      - Cesta nejlepší světlušky v každé generaci
      - Start (modrý bod), konec (zelená hvězda)
    
    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi [(low, high), ...].
    pop_size : int
        Počet světlušek v populaci (typicky 15-40).
    beta_0 : float
        Maximální atraktivita β₀ při r=0 (typicky 1.0).
    alpha : float
        Parametr náhodného pohybu α (typicky 0.1-0.5).
    max_generations : int
        Maximální počet generací (iterací).
    seed : Optional[int]
        Seed pro reprodukovatelnost výsledků.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D funkce).
    num_points : int
        Hustota mřížky pro heatmapu vizualizace.
    
    Návrat
    ------
    (best_position, best_fitness) : Tuple[List[float], float]
        Nejlepší nalezené řešení a jeho fitness hodnota.
    """
    
    # Vytvoříme generátor náhodných čísel s daným seedem (pro reprodukovatelnost)
    rng = random.Random(seed)
    
    # Počet dimenzí (např. pro 2D funkci je dim=2)
    dim = len(bounds)
    
    # Pomocná třída pro reprezentaci světlušky
    # Každá světluška má pozici v prostoru a intenzitu světla (fitness hodnotu)
    class Firefly:
        def __init__(self, position: List[float]):
            self.position = position  # Aktuální pozice světlušky v prostoru
            self.fitness = float('inf')  # Fitness hodnota (intenzita světla) - čím nižší, tím jasnější
    
    # Pomocná funkce pro výpočet Euklidovské vzdálenosti mezi dvěma světluškami
    def euclidean_distance(pos1: List[float], pos2: List[float]) -> float:
        """
        Vypočítá Euklidovskou vzdálenost mezi dvěma pozicemi.
        r_ij = sqrt(sum((x_i - x_j)^2))
        """
        distance = 0.0
        for i in range(dim):
            distance += (pos1[i] - pos2[i]) ** 2
        return math.sqrt(distance)
    
    # Pomocná funkce: clamp (oříznutí) hodnoty do mezí
    # Zajišťuje, že světluška nevyletí mimo funcki
    def clamp(value: float, low: float, high: float) -> float:
        """Ořízne hodnotu do intervalu [low, high]."""
        if value < low:
            return low
        elif value > high:
            return high
        else:
            return value
    
    # 1) Inicializace populace světlušek
    # Vytvoříme pop_size světlušek s náhodnými pozicemi v daných mezích
    population: List[Firefly] = []
    first_firefly_position = None  # Uložíme první pozici pro vizualizaci startu
    
    for idx in range(pop_size):
        # Náhodná počáteční pozice v mezích
        position = []
        for low, high in bounds:
            # Pozice v rozsahu [low, high] z rovnoměrného rozdělení
            pos = rng.uniform(low, high)
            position.append(pos)
        
        # Vytvoříme světlušku s touto pozicí
        firefly = Firefly(position)
        
        # Vyhodnotíme počáteční fitness (intenzitu světla)
        # Pro minimalizaci: čím nižší fitness, tím jasnější světluška
        firefly.fitness = objective(position)
        
        # Přidáme světlušku do populace
        population.append(firefly)
        
        # Uložíme první pozici jako reprezentant počátečního stavu
        if idx == 0:
            first_firefly_position = list(position)
    
    # Najdeme nejlepší světlušku v počáteční populaci (nejnižší fitness = nejjasnější)
    best_firefly = min(population, key=lambda f: f.fitness)
    best_position = list(best_firefly.position)
    best_fitness = best_firefly.fitness
    
    # Pro vizualizaci: ukládáme cestu nejlepší světlušky v každé generaci
    path: List[Tuple[List[float], float]] = [
        (first_firefly_position, objective(first_firefly_position)),
        (list(best_position), best_fitness)
    ]
    
    # 2) Hlavní smyčka - generace Firefly algoritmu
    # V každé generaci se světlušky pohybují směrem k jasnějším světluškám
    for generation in range(max_generations):
        # Pro každou světlušku i v populaci
        for i in range(pop_size):
            firefly_i = population[i]
            
            # Porovnáme světlušku i se všemi ostatními světluškami j
            for j in range(pop_size):
                firefly_j = population[j]
                
                # ================================================================
                # KLÍČOVÁ PODMÍNKA FIREFLY ALGORITMU:
                # Světluška i se pohybuje směrem k světlušce j pouze pokud
                # je j jasnější než i, tj. má lepší (nižší) fitness hodnotu
                # ================================================================
                # Light intensity I_i je určeno fitness hodnotou f(x_i)
                # Pro minimalizaci: I_j > I_i znamená f(x_j) < f(x_i)
                if firefly_j.fitness < firefly_i.fitness:
                    # Světluška j je jasnější než i, proto i bude přitahována k j
                    
                    # Vypočítáme Euklidovskou vzdálenost r_ij mezi světluškami i a j
                    r = euclidean_distance(firefly_i.position, firefly_j.position)
                    
                    # Vypočítáme atraktivitu β (beta) v závislosti na vzdálenosti
                    # Používáme zjednodušený vzorec:
                    #   β = β₀ / (1 + r)
                    # místo složitějšího:
                    #   β = β₀ * exp(-γ * r²)
                    # Tento zjednodušený vzorec eliminuje potřebu ladit γ
                    beta = beta_0 / (1.0 + r)
                    
                    # ================================================================
                    # HLAVNÍ ROVNICE POHYBU SVĚTLUŠKY:
                    # ================================================================
                    # x_i(t+1) = x_i(t) + β * (x_j(t) - x_i(t)) + α * ε
                    #
                    # kde:
                    #   x_i(t)    = aktuální pozice světlušky i
                    #   x_j(t)    = pozice jasnější světlušky j
                    #   β         = atraktivita (klesá se vzdáleností)
                    #   α         = parametr náhodného pohybu
                    #   ε         = náhodný vektor z normálního rozdělení N(0,1)
                    #
                    # První člen: β * (x_j - x_i) = přitažlivost k jasnější světlušce
                    # Druhý člen: α * ε = náhodný pohyb (exploration)
                    # ================================================================
                    
                    # Vytvoříme novou pozici pro světlušku i
                    new_position = []
                    for d in range(dim):
                        # Směrový vektor k jasnější světlušce j
                        attraction = firefly_j.position[d] - firefly_i.position[d]
                        
                        # Náhodná složka z normálního rozdělení N(0, 1)
                        # !Používáme NORMÁLNÍ ROZDĚLENÍ (Gaussovo), ne rovnoměrné. Narozdíl od jiných algoritmů
                        epsilon = rng.gauss(0.0, 1.0)  # Normální rozdělení: μ=0, σ=1
                        
                        # Nová souřadnice podle hlavní rovnice
                        # = aktuální pozice + přitažlivost + náhodný pohyb
                        new_coord = firefly_i.position[d] + beta * attraction + alpha * epsilon
                        
                        # Ošetření hranic - pokud světluška vylétne mimo funkci,
                        # vrátíme ji zpět do hranic
                        low, high = bounds[d]
                        new_coord = clamp(new_coord, low, high)
                        
                        new_position.append(new_coord)
                    
                    # Aktualizujeme pozici světlušky i
                    firefly_i.position = new_position
                    
                    # Vyhodnotíme novou fitness hodnotu (intenzitu světla) na nové pozici
                    # Tato hodnota se použije v dalších porovnáních v této generaci
                    firefly_i.fitness = objective(new_position)
        
        # Po dokončení všech pohybů v této generaci najdeme aktuálně nejlepší světlušku
        # (světlušku s nejnižší fitness hodnotou = nejjasnější)
        current_best = min(population, key=lambda f: f.fitness)
        
        # Pokud je aktuálně nejlepší světluška lepší než dosud nalezené globální best,
        # aktualizujeme globální best
        if current_best.fitness < best_fitness:
            best_position = list(current_best.position)
            best_fitness = current_best.fitness
        
        # Uložíme aktuální nejlepší pozici pro vizualizaci
        path.append((list(best_position), best_fitness))
    
    # 3) Finální výsledek
    # Po dokončení všech generací vrátíme nejlepší nalezené řešení
    final_best = min(population, key=lambda f: f.fitness)
    best_position = final_best.position
    best_fitness = final_best.fitness
    
    # 4) Vizualizace (heatmapa pro 2D)
    # Pokud je požadována vizualizace a funkce je 2D, vytvoříme heatmapu
    if visualize and dim == 2:
        import matplotlib.pyplot as plt
        
        # Vytvoříme mřížku a vyhodnotíme cílovou funkci pro heatmapu
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)
        
        # Vytvoříme figure a axes pro vykreslení
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Heatmapa: vyplněné kontury zobrazující hodnoty cílové funkce
        levels = 50  # Počet úrovní barev
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar (legendu barev) pro zobrazení hodnot funkce
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)
        
        # Volitelně: přidáme contour čáry (izolinie) pro lepší čitelnost
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        
        # Popisky os
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Firefly Algorithm – Heatmap')
        
        # Vykreslíme cestu nejlepší světlušky přes všechny generace
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako bílá čára
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='Best v čase')
            
            # Zvýrazníme body, kde došlo ke zlepšení (fitness klesla)
            improvement_indices = [0]
            for i in range(1, len(path)):
                if path[i][1] < path[i-1][1]:  # Pokud je fitness lepší než předchozí
                    improvement_indices.append(i)
            
            # Vykreslíme body zlepšení žlutými kroužky
            if improvement_indices:
                imp_x = [path_x[i] for i in improvement_indices]
                imp_y = [path_y[i] for i in improvement_indices]
                ax.scatter(imp_x, imp_y, s=80, c='yellow', edgecolor='white',
                          linewidth=1.5, marker='o', zorder=6,
                          label=f'Zlepšení ({len(improvement_indices)}x)')
            
            # Start bod (modrý kruh) - první světluška v populaci
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue',
                      edgecolor='white', linewidth=2, marker='o',
                      label='Start', zorder=5)
            
            # Finální nejlepší bod (zelená hvězda)
            ax.scatter([best_position[0]], [best_position[1]], s=200, c='lime',
                      edgecolor='white', linewidth=2, marker='*',
                      label='Best', zorder=5)
        
        # Legenda s popisem
        ax.legend(loc='upper right')
        
        # Mřížka pro lepší orientaci
        ax.grid(True, alpha=0.3)
        
        # Textový box s informacemi o výsledku
        msg = f"Firefly: best=({best_position[0]:.4f}, {best_position[1]:.4f}), f={best_fitness:.6g}, gen={max_generations}, pop={pop_size}, α={alpha}, β₀={beta_0}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))
        
        # Úprava rozložení a zobrazení grafu
        plt.tight_layout()
        plt.show()
    
    return best_position, best_fitness

def soma_all_to_one(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 10,  # velikost populace
    path_length: float = 3.0,  # PRT - délka cesty k vůdci
    step: float = 0.11,  # délka kroku
    prt: float = 0.3,  # perturbace - pravděpodobnost změny dimenze - generuje se z něj perturbační vektor
    max_migrations: int = 20,  # počet migračních kol (iterací)
    seed: Optional[int] = None,  # náhodný seed
    visualize: bool = False,  # vizualizace
    num_points: int = 200,  # hustota mřížky pro heatmapu
) -> Tuple[List[float], float]:
    """
    SOMA (Self-Organizing Migrating Algorithm) - varianta AllToOne.
    
    Algoritmus:
      1. Inicializujeme populaci jedinců s náhodnými pozicemi v daných mezích
      2. Pro každou migraci (iterace):
         - Najdeme leadera (nejlepšího jedince v populaci)
         - Pro každého jedince (kromě leadera):
           a) Vytvoříme PRT vektor (perturbační vektor) - náhodná binární maska
           b) Jedinec se pohybuje směrem k leaderovi v krocích
           c) Pro každý krok t (t ∈ {0, step, 2*step, ..., path_length}):
              - Nová pozice = aktuální + t * PRT * (leader - aktuální)
              - Vyhodnotíme fitness v nové pozici
           d) Jedinec přeskočí na nejlepší pozici z jeho cesty
      3. Vrátíme nejlepší nalezené řešení
    
    Parametry algoritmu:
      - path_length: určuje jak daleko se jedinec může dostat od leadera (typicky 1.5-3.0)
        * Vyšší hodnota = větší exploration
      - step: velikost kroku po cestě (typicky 0.11-0.33)
        * Menší step = více kroků = přesnější hledání, ale pomalejší
      - prt: pravděpodobnost pro PRT vektor (typicky 0.1-0.4)
        * Určuje kolik dimenzí se bude měnit při migraci
    
    Vizualizace (jen 2D):
      - Heatmapa funkce na pozadí
      - Cesta nejlepšího jedince (leadera) v každé migraci
      - Start (modrý bod), konec (zelená hvězda)
    
    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi.
    pop_size : int
        Velikost populace.
    path_length : float
        Parametr PathLength - jak daleko směrem k leaderovi (typicky 1.1 - 5>).
    step : float
        Velikost kroku (typicky 0.11 - path_length).
        Vzorkování - udává jak hustě jedinec má "skákat" po trajektorii path_length.
    prt : float
        Perturbační parametr - pravděpodobnost změny dimenze (typicky 0.1-0.4, ale pohybuje se v intervalu [0,1]).
        Jakési rušení jedince, ekvivalent mutace, vzniklo ryze jako geometrická záležitost.
        Má dopad na dráhu jedince.
    max_migrations : int
        Počet migrací (iterací).
    seed : Optional[int]
        Seed pro reprodukovatelnost.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D).
    num_points : int
        Hustota mřížky pro heatmapu.
    
    Návrat
    ------
    (best_x, best_f) : Tuple[List[float], float]
        Nejlepší nalezené řešení a jeho hodnota.
    """
    import random
    
    # Vytvoříme generátor náhodných čísel s daným seedem (pro reprodukovatelnost)
    rng = random.Random(seed)
    
    # Počet dimenzí (např. pro 2D funkci je dim=2)
    dim = len(bounds)
    
    # Pomocná třída pro reprezentaci jedince v populaci
    # Na rozdíl od PSO, kde částice má pozici, rychlost a osobní nejlepší pozici (pbest),
    # jedinec v SOMA má pouze aktuální pozici a fitness (žádné pbest, žádnou rychlost)
    class Individual:
        def __init__(self, position: List[float]):
            self.position = position  # Aktuální pozice jedince v prostoru řešení
            self.fitness = float('inf')  # Fitness (hodnota cílové funkce) - inicializujeme na nekonečno
    
    # Pomocná funkce: clamp (oříznutí) hodnoty do mezí, abychom nevyletěli z definičního oboru funkce
    def clamp(value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value
    
    # 1) Inicializace populace jedinců
    # Vytvoříme pop_size jedinců s náhodnými pozicemi v daných mezích
    population: List[Individual] = []
    first_individual_position = None  # Uložíme první pozici pro vizualizaci startu
    
    for idx in range(pop_size):
        # dimenzí je tolik, kolik máme hranic
        # pro každou dimenzi jeden prvek v position
        position = []
        
        # Náhodná počáteční pozice v mezích
        for low, high in bounds:
            # Pozice v rozsahu [low, high]
            pos = rng.uniform(low, high)  # uniform vrací náhodný float z rovnoměrného rozdělení
            position.append(pos)
        
        # Vytvoříme jedince s touto pozicí
        individual = Individual(position)
        
        # Vyhodnotíme počáteční fitness (hodnotu cílové funkce)
        individual.fitness = objective(position)  # objective je předaná cílová funkce, např. sphere
        
        # Přidáme jedince do populace
        population.append(individual)
        
        # Uložíme první pozici jako reprezentant počátečního stavu
        if idx == 0:
            first_individual_position = list(position)
    
    # Najdeme počátečního leadera (nejlepšího jedince z počáteční populace)
    # min() s key=lambda najde jedince s minimální hodnotou fitness
    leader = min(population, key=lambda ind: ind.fitness)
    leader_position = list(leader.position)  # Kopie pozice leadera
    leader_fitness = leader.fitness  # Fitness hodnota leadera
    
    # Pro vizualizaci: ukládáme cestu leadera v každé migraci
    # Začínáme s první pozicí (ne nejlepší), aby bylo vidět skutečný start
    path: List[Tuple[List[float], float]] = [
        (first_individual_position, objective(first_individual_position)),
        (list(leader_position), leader_fitness)
    ]
    
    # 2) Hlavní smyčka - migrace SOMA
    # V každé migraci se všichni jedinci (kromě leadera) pohybují směrem k leaderovi
    for migration in range(max_migrations):
        # Na začátku každé migrace najdeme aktuálního leadera (nejlepšího jedince v populaci)
        # Leader se mohl změnit, protože jedinci se pohybovali v předchozí migraci
        leader = min(population, key=lambda ind: ind.fitness)
        
        # Pro každého jedince (kromě leadera) provedeme migraci směrem k leaderovi
        for individual in population:
            # DŮLEŽITÉ: Leader se nepohybuje, zůstává na místě jako cíl pro ostatní
            # Kontrolujeme identitu objektu (is), ne pouze hodnotu (==)
            if individual is leader:
                continue  # Přeskočíme leadera a jdeme na dalšího jedince
            
            # Vytvoříme PRT vektor (Perturbation vector - perturbační vektor)
            # Je to binární maska (vektor nul a jedniček), která určuje které dimenze se budou měnit
            # Funguje jako "filtr" - pokud je PRT[i] = 0, dimenze i se nemění, pokud je 1, mění se
            prt_vector = []
            for d in range(dim):
                # Pro každou dimenzi s pravděpodobností prt nastavíme 1 (dimenze se bude měnit)
                # s pravděpodobností (1-prt) nastavíme 0 (dimenze zůstane nezměněna)
                # Příklad: pokud prt=0.3, tak průměrně 30% dimenzí se bude měnit
                prt_vector.append(1.0 if rng.random() < prt else 0.0)
            
            # Ošetření speciálního případu: pokud je celý PRT vektor nulový (všechny dimenze = 0)
            # jedinec by se vůbec nepohnul. Proto nastavíme alespoň jednu dimenzi na 1
            if all(v == 0.0 for v in prt_vector):
                random_dim = rng.randint(0, dim - 1)  # Vybereme náhodnou dimenzi
                prt_vector[random_dim] = 1.0  # A nastavíme ji na 1
            
            # Jedinec se pohybuje směrem k leaderovi v malých krocích
            # Budeme si pamatovat nejlepší nalezenou pozici na celé cestě
            best_position_on_path = list(individual.position)  # Začínáme s aktuální pozicí
            best_fitness_on_path = individual.fitness  # A její fitness hodnotou
            
            # Parametr t (krok cesty) jde od 0 do path_length s krokem step
            # Příklad: pokud path_length=3.0 a step=0.11, pak t bude: 0, 0.11, 0.22, 0.33, ..., 2.97, 3.0
            # Čím menší step, tím více kroků a přesnější prohledávání (ale pomalejší)
            t = 0.0
            while t <= path_length:
                # ================================================================
                # SRDCE SOMA - ZÁKLADNÍ ROVNICE MIGRACE:
                # ================================================================
                # r = r0 + m * t * PRTVector
                #
                # kde:
                #   r         = nová pozice jedince
                #   r0        = aktuální pozice jedince (startovní bod)
                #   m         = (leader - r0) = směrový vektor k leaderovi
                #   t         = parametr cesty (0 až path_length)
                #   PRTVector = perturbační vektor (binární maska 0/1)
                #   *         = násobení po složkách (element-wise)
                #
                # V implementaci: new_position = individual.position + t * prt_vector * (leader.position - individual.position)
                # ================================================================
                
                # Vypočítáme novou pozici pro tento krok t
                new_position = []
                for d in range(dim):
                    # Směrový vektor k leaderovi (rozdíl mezi pozicí leadera a jedince)
                    # Říká "kterým směrem a jak daleko je leader"
                    direction = leader.position[d] - individual.position[d]
                    
                    # Nová souřadnice s aplikací PRT vektoru
                    # t určuje jak daleko po cestě jsme (0 = start, path_length = maximálně daleko)
                    # prt_vector[d] určuje zda se tato dimenze vůbec mění (0 = ne, 1 = ano)
                    # Příklad: pokud prt_vector[d]=0, pak direction se vynásobí 0 a dimenze zůstane stejná
                    new_coord = individual.position[d] + t * prt_vector[d] * direction
                    
                    # Ošetření hranic - pokud jedinec vylétne mimo povolený prostor, vrátíme ho zpět
                    low, high = bounds[d]
                    new_coord = clamp(new_coord, low, high)
                    
                    new_position.append(new_coord)
                
                # Vyhodnotíme fitness (cílovou funkci) v této nové pozici na cestě
                new_fitness = objective(new_position)
                
                # Pokud je tato pozice lepší (menší fitness) než dosud nejlepší na celé cestě
                # uložíme si ji jako novou nejlepší pozici na cestě
                if new_fitness < best_fitness_on_path:
                    best_fitness_on_path = new_fitness
                    best_position_on_path = list(new_position)  # Kopie pozice
                
                # Posuneme se o krok step dále po cestě
                t += step
            
            # KLÍČOVÝ KROK SOMA: Po projití celé cesty jedinec "přeskočí" (teleportuje se)
            # na nejlepší nalezenou pozici z celé jeho cesty
            # Nejedná se o plynulý pohyb jako u PSO, ale o diskrétní skok
            individual.position = best_position_on_path
            individual.fitness = best_fitness_on_path
        
        # Po skončení migrace (když se všichni jedinci kromě leadera přemístili)
        # aktualizujeme leadera - najdeme nového nejlepšího jedince
        # Může to být stále stejný jedinec, nebo se mohl změnit
        leader = min(population, key=lambda ind: ind.fitness)
        
        # Uložíme pozici aktuálního leadera pro vizualizaci
        path.append((list(leader.position), leader.fitness))
    
    # 3) Finální výsledek
    # Po dokončení všech migrací najdeme finálně nejlepšího jedince v celé populaci
    best_individual = min(population, key=lambda ind: ind.fitness)
    best_position = best_individual.position  # Jeho pozice je výsledné řešení
    best_fitness = best_individual.fitness  # A jeho fitness je hodnota cílové funkce v tomto řešení
    
    # 4) Vizualizace (heatmapa pro 2D)
    if visualize and dim == 2:
        import matplotlib.pyplot as plt
        
        # Vytvoříme mřížku a vyhodnotíme funkci
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Heatmapa: vyplněné kontury
        levels = 50
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)
        
        # Volitelně: přidáme contour čáry (izolinie)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('SOMA AllToOne – Heatmap')
        
        # Vykreslíme cestu leadera
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako čára
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='Leader v čase')
            
            # Zvýrazníme body, kde došlo ke zlepšení
            improvement_indices = [0]
            for i in range(1, len(path)):
                if path[i][1] < path[i-1][1]:
                    improvement_indices.append(i)
            
            if improvement_indices:
                imp_x = [path_x[i] for i in improvement_indices]
                imp_y = [path_y[i] for i in improvement_indices]
                ax.scatter(imp_x, imp_y, s=80, c='yellow', edgecolor='white',
                          linewidth=1.5, marker='o', zorder=6,
                          label=f'Zlepšení ({len(improvement_indices)}x)')
            
            # Start bod (modrý)
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue',
                      edgecolor='white', linewidth=2, marker='o',
                      label='Start', zorder=5)
            
            # Finální bod (zelený)
            ax.scatter([best_position[0]], [best_position[1]], s=200, c='lime',
                      edgecolor='white', linewidth=2, marker='*',
                      label='Best', zorder=5)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        msg = f"SOMA: best=({best_position[0]:.4f}, {best_position[1]:.4f}), f={best_fitness:.6g}, migrations={max_migrations}, pop={pop_size}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))
        
        plt.tight_layout()
        plt.show()
    
    return best_position, best_fitness

def particle_swarm_optimization(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    num_particles: int = 15, #pop_size
    w: float = 0.7, # inertia weight - setrvačná váha
    c1: float = 2.0, # cognitive parameter - kognitívní koeficient
    c2: float = 2.0, # social parameter - sociální koeficient
    max_iter: int = 50, # maximální počet iterací
    seed: Optional[int] = None, # náhodný seed pro reprodukovatelnost
    visualize: bool = False, # má-li se vytvořit heatmap vizualizace
    num_points: int = 200, # hustota mřížky pro heatmapu
) -> Tuple[List[float], float]:
    """
    Particle Swarm Optimization (PSO) s inertia weight - algoritmus pro minimalizaci.

    Algoritmus:
      1. Inicializujeme roj částic s náhodnými pozicemi a rychlostmi v daných mezích
      2. Pro každou iteraci (až do max_iter):
         - Pro každou částici:
           a) Vyhodnotíme fitness (cílovou funkci) v aktuální pozici
           b) Aktualizujeme osobní nejlepší pozici (pbest) pokud je aktuální lepší
           c) Aktualizujeme globální nejlepší pozici (gbest) pokud je některá pbest lepší
         - Pro každou částici aktualizujeme rychlost a pozici:
           d) v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
           e) x_i = x_i + v_i
           f) Ošetříme hranice (clamp)
      3. Vrátíme globální nejlepší řešení (gbest)

    Parametry algoritmu:
      - w (inertia weight): setrvačná váha - řídí vliv předchozí rychlosti
        * Vyšší w (např. 0.9) znamená větší exploraci (průzkum)
        * Nižší w (např. 0.4) znamená větší exploitation (využití známých řešení)
      - c1 (cognitive parameter): ovlivňuje táhnutí k osobnímu nejlepšímu
      - c2 (social parameter): ovlivňuje táhnutí ke globálnímu nejlepšímu
      - r1, r2: náhodná čísla v [0, 1] - zajišťují stochastický charakter

    Vizualizace (jen 2D):
      - Heatmapa funkce na pozadí
      - Cesta globálně nejlepší částice v každé iteraci (bílá čára)
      - Start (modrý bod), konec (zelená hvězda)

    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi.
    num_particles : int
        Počet částic v roji.
    w : float
        Inertia weight (setrvačná váha), typicky 0.4 - 0.9.
    c1 : float
        Cognitive parameter (kognitívní koeficient) - vliv osobního nejlepšího.
    c2 : float
        Social parameter (sociální koeficient) - vliv globálního nejlepšího.
    max_iter : int
        Maximální počet iterací.
    seed : Optional[int]
        Seed pro reprodukovatelnost.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D).
    num_points : int
        Hustota mřížky pro heatmapu.

    Návrat
    ------
    (best_x, best_f) : Tuple[List[float], float]
        Nejlepší nalezené řešení a jeho hodnota.
    """
    import random
    
    rng = random.Random(seed)
    dim = len(bounds)

    # Pomocná třída pro reprezentaci částice
    class Particle:
        def __init__(self, position: List[float], velocity: List[float]):
            self.position = position             # Aktuální pozice částice
            self.velocity = velocity             # Aktuální rychlost částice
            self.best_position = list(position)  # Osobní nejlepší pozice (pbest) dané částice
            self.best_value = float('inf')       # Hodnota fitness v pbest
            self.current_value = float('inf')    # Aktuální hodnota fitness

    # Pomocná funkce: clamp hodnoty do mezí, abychom nevyletěli z funkce
    def clamp(value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value

    # 1) Inicializace roje částic
    # Vytvoříme num_particles částic s náhodnými pozicemi a rychlostmi
    particles: List[Particle] = []
    first_particle_position = None  # Uložíme první částici pro vizualizaci startu
    
    for idx in range(num_particles):
        # dimenzí je tolik, kolik máme hranic
        # pro každou dimenzi jeden prvek v position a velocity
        position = []
        velocity = []
        # Náhodná počáteční pozice v mezích
        for low, high in bounds:
            # Pozice v rozsahu [low, high]
            pos = rng.uniform(low, high)
            position.append(pos)
            
            # Rychlost inicializujeme v rozsahu [-(high-low)/2, (high-low)/2]
            # To zabrání příliš velkým počátečním rychlostem
            v_range = (high - low) / 2.0 # polovina rozsahu
            vel = rng.uniform(-v_range, v_range) # na rychlost použijeme funkci uniform, která v daném rozsahu generuje rovnoměrně rozložený náhodný float
            velocity.append(vel) # přidáme rychlost do seznamu velocity
        
        # Vytvoříme částici
        particle = Particle(position, velocity)
        
        # Vyhodnotíme počáteční fitness
        particle.current_value = objective(position) # objective je předaná cílová funkce, např. sphere
        particle.best_value = particle.current_value # nejlepší hodnotu částice inicializujeme na aktuální hodnotu
        particle.best_position = list(position) # nejlepší pozici částice inicializujeme na aktuální pozici
        
        particles.append(particle) # přidáme plně inicializovanou částici do seznamu částic
        
        # Uložíme první částici jako reprezentant počátečního stavu
        if idx == 0:
            first_particle_position = list(position)

    # Najdeme globální nejlepší pozici (gbest) z počáteční populace

    # gbest jsou globální nejlepší pozice a hodnota pro celý roj částic (jednu populaci)
    gbest_position = list(particles[0].best_position)
    gbest_value = particles[0].best_value
    
    # projdeme všechny částice z počáteční populace a najdeme nejlepšího z nich
    for particle in particles:
        if particle.best_value < gbest_value:
            gbest_value = particle.best_value
            gbest_position = list(particle.best_position)

    # Pro vizualizaci: ukládáme cestu gbest v každé iteraci
    # Začínáme s první částicí (ne nejlepší), aby bylo vidět skutečný start
    path: List[Tuple[List[float], float]] = [(first_particle_position, objective(first_particle_position))]
    # Přidáme i gbest z počáteční populace
    path.append((list(gbest_position), gbest_value))

    # 2) Hlavní smyčka - iterace PSO
    for iteration in range(max_iter):
        # Jako první krok vyhodnotíme pro každou iteraci všechny částice (jejich fitness)
        for particle in particles:
            # Vyhodnotíme fitness v aktuální pozici
            # .current_value je aktuální hodnota fitness částice
            particle.current_value = objective(particle.position) # zavoláme funkci objective s aktuální pozicí částice
            
            # Aktualizace osobního nejlepšího (pbest) dané částice
            if particle.current_value < particle.best_value:
                particle.best_value = particle.current_value
                particle.best_position = list(particle.position)
                
                # Aktualizace globálního nejlepšího (gbest)
                if particle.best_value < gbest_value:
                    gbest_value = particle.best_value
                    gbest_position = list(particle.best_position)
        
        # HLAVNÍ PODCYKLUS KAŽDÉ ITERACE:
        # Aktualizace rychlostí a pozic všech částic
        for particle in particles:
            # pro každou částici je třeba vyhodnotit každou dimenzi zvlášť
            for d in range(dim):
                # Náhodné koeficienty pro stochastický charakter
                r1 = rng.random()  # Náhodné číslo v rozsahu [0, 1]
                r2 = rng.random()  # Náhodné číslo v rozsahu [0, 1]

                # připomínka významu proměnných ve vzorci:
                # w - inertia weight (setrvačná váha), částice má momentum, w určí jak moc má částice pokračovat směrem své předchozí rychlosti
                # c1 - cognitive parameter (kognitívní koeficient), určuje jak moc se částice řídí svým osobním nejlepším (pbest)
                # c2 - social parameter (sociální koeficient), určuje jak moc se částice řídí globálním nejlepším (gbest)

                # Aktualizace rychlosti podle PSO vzorce:
                # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
                inertia = w * particle.velocity[d]
                cognitive = c1 * r1 * (particle.best_position[d] - particle.position[d])
                social = c2 * r2 * (gbest_position[d] - particle.position[d])
                
                # máme spočítány všechny tři složky, sečteme je a uložíme jako novou rychlost částice pro danou dimenzi
                particle.velocity[d] = inertia + cognitive + social
                
                # Aktualizace pozice: x = x + v (čili pozice se posune o rychlost v dané dimenzi)
                particle.position[d] = particle.position[d] + particle.velocity[d]
                
                # Ošetření hranic - pokud částice vylétne mimo, vrátíme ji zpět
                low, high = bounds[d]
                particle.position[d] = clamp(particle.position[d], low, high)
                
                # Pokud částice narazila na hranici, můžeme volitelně zpomalit rychlost
                # (aby se nenalepila na hranici)
                if particle.position[d] == low or particle.position[d] == high:
                    particle.velocity[d] *= -0.5  # Odraz s tlumením
        
        # Uložíme gbest této iterace pro vizualizaci
        path.append((list(gbest_position), gbest_value))

    # 3) Vizualizace (heatmapa pro 2D)
    if visualize and dim == 2:
        import matplotlib.pyplot as plt

        # Vytvoříme mřížku a vyhodnotíme funkci
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Heatmapa: vyplněné kontury
        levels = 50
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)

        # Volitelně: přidáme contour čáry (izolinie)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Particle Swarm Optimization (PSO) – Heatmap')

        # Vykreslíme cestu gbest
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako čára - zobrazí VŠECHNY iterace (i když gbest zůstává stejný)
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='Gbest v čase')
            
            # Zvýrazníme body, kde došlo ke zlepšení (změna gbest hodnoty)
            improvement_indices = [0]  # První bod vždy
            for i in range(1, len(path)):
                if path[i][1] < path[i-1][1]:  # Zlepšení fitness
                    improvement_indices.append(i)
            
            if improvement_indices:
                imp_x = [path_x[i] for i in improvement_indices]
                imp_y = [path_y[i] for i in improvement_indices]
                ax.scatter(imp_x, imp_y, s=80, c='yellow', edgecolor='white',
                          linewidth=1.5, marker='o', zorder=6, 
                          label=f'Zlepšení ({len(improvement_indices)}x)')
            
            # Start bod (modrý)
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue', 
                      edgecolor='white', linewidth=2, marker='o', 
                      label='Start', zorder=5)
            
            # Finální bod (zelený)
            ax.scatter([gbest_position[0]], [gbest_position[1]], s=200, c='lime', 
                      edgecolor='white', linewidth=2, marker='*', 
                      label='Best', zorder=5)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        msg = f"PSO: best=({gbest_position[0]:.4f}, {gbest_position[1]:.4f}), f={gbest_value:.6g}, iter={max_iter}, particles={num_particles}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))

        plt.tight_layout()
        plt.show()

    return gbest_position, gbest_value

def differential_evolution(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    NP: int = 20,
    F: float = 0.5,
    CR: float = 0.5,
    max_gen: int = 50,
    seed: Optional[int] = None,
    visualize: bool = False,
    num_points: int = 200,
) -> Tuple[List[float], float]:
    """
    Differential Evolution (DE/rand/1/bin) - evolučníalgoritmus pro minimalizaci.

    Algoritmus:
      1. Vygenerujeme populaci NP jedinců (náhodné vektory v daných mezích)
      2. Pro každou generaci (až do max_gen):
         - Pro každý jedinec (tzv. target vektor x_i):
           a) Vybereme 3 náhodné různé jedince: x_r1, x_r2, x_r3
           b) Vytvoříme mutační vektor: v = x_r3 + F * (x_r1 - x_r2)
           c) Křížením vytvoříme trial vektor u (kombinace v a x_i podle CR)
           d) Pokud je u lepší nebo rovno x_i, nahradíme x_i vektorem u
      3. Vrátíme nejlepšího jedince z finální populace

    Vizualizace (jen 2D):
      - Heatmapa funkce na pozadí
      - Cesta nejlepšího jedince v každé generaci (bílá čára)
      - Start (modrý bod), konec (zelená hvězda)

    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi.
    NP : int
        Velikost populace (počet jedinců).
    F : float
        Mutační konstanta (scaling factor), typicky 0.4 - 1.0.
    CR : float
        Pravděpodobnost crossoveru, rozsah [0, 1].
    max_gen : int
        Maximální počet generací.
    seed : Optional[int]
        Seed pro reprodukovatelnost.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D).
    num_points : int
        Hustota mřížky pro heatmapu.

    Návrat
    ------
    (best_x, best_f) : Tuple[List[float], float]
        Nejlepší nalezené řešení a jeho hodnota.
    """
    import random
    import copy
    
    rng = random.Random(seed)
    dim = len(bounds)

    # Pomocná třída pro reprezentaci jedince (řešení)
    class Individual:
        def __init__(self, params: List[float], f_value: float):
            self.params = params  # vektor parametrů
            self.f = f_value      # hodnota fitness (cílové funkce)

    # Pomocná funkce: clamp hodnoty do mezí
    def clamp(value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value

    # 1) Inicializace populace
    # Vygenerujeme NP náhodných jedinců v zadaných mezích
    population: List[Individual] = []
    first_individual_params = None  # Uložíme prvního jedince pro vizualizaci startu
    
    for idx in range(NP):
        params = []
        for low, high in bounds:
            value = rng.uniform(low, high)
            params.append(value)
        f_value = objective(params)
        population.append(Individual(params, f_value))
        
        # Uložíme prvního jedince jako reprezentanta počátečního stavu
        if idx == 0:
            first_individual_params = list(params)

    # Najdeme nejlepšího jedince v počáteční populaci
    best_idx = 0
    for i in range(1, NP):
        if population[i].f < population[best_idx].f:
            best_idx = i
    
    best_x = list(population[best_idx].params)
    best_f = float(population[best_idx].f)

    # Pro vizualizaci: ukládáme cestu nejlepšího jedince v každé generaci
    # Začínáme s prvním vygenerovaným jedincem (ne nejlepším), aby bylo vidět skutečný start
    path: List[Tuple[List[float], float]] = [(first_individual_params, objective(first_individual_params))]
    # Přidáme i nejlepšího z počáteční populace
    path.append((list(best_x), best_f))

    # 2) Hlavní smyčka - evoluce přes generace
    for g in range(max_gen):
        # Vytvoříme novou populaci (deep copy staré)
        new_population = copy.deepcopy(population)

        # Pro každého jedince v populaci
        for i in range(NP):
            # Cílový vektor (target vector)
            x_i = population[i]

            # a) Vybereme 3 náhodné různé indexy r1, r2, r3
            # Musí platit: r1 != r2 != r3 != i
            indices = list(range(NP))
            indices.remove(i)  # odstraníme index aktuálního jedince
            
            # Vybereme 3 různé indexy
            r1, r2, r3 = rng.sample(indices, 3)

            # b) Mutace: v = x_r3 + F * (x_r1 - x_r2)
            v = []
            for j in range(dim):
                low, high = bounds[j]
                # Mutační vektor podle vzorce
                v_j = population[r3].params[j] + F * (population[r1].params[j] - population[r2].params[j])
                # Ošetření hranic
                v_j = clamp(v_j, low, high)
                v.append(v_j)

            # c) Křížení (crossover): vytvoření trial vektoru u
            u = [0.0] * dim
            # j_rnd zajistí, že alespoň jeden parametr bude z mutačního vektoru
            # a to tak, že j_rnd je jeden z indexů dimenzí
            j_rnd = rng.randint(0, dim - 1)

            # CR je pravděpodobnost, že vezmeme parametr z mutačního vektoru
            # pro každou dimenzi rozhodneme podle CR
            for j in range(dim):
                # pokud je náhodné číslo menší než CR nebo je to j_rnd, vezmeme z mutačního vektoru v
                if rng.random() < CR or j == j_rnd:
                    # Vezmeme parametr z mutačního vektoru
                    u[j] = v[j]
                else:
                    # Vezmeme parametr z původního jedince
                    u[j] = x_i.params[j]

            # d) Vyhodnocení trial vektoru
            f_u = objective(u)

            # Selekce: pokud je trial vektor lepší nebo roven, nahradíme jím cílový vektor
            if f_u <= x_i.f:
                new_population[i] = Individual(list(u), f_u)
                
                # Aktualizace globálního nejlepšího
                if f_u < best_f:
                    best_x = list(u)
                    best_f = float(f_u)

        # Nahradíme starou populaci novou
        population = new_population
        
        # Uložíme nejlepšího jedince této generace pro vizualizaci
        path.append((list(best_x), best_f))

    # 3) Vizualizace (heatmapa pro 2D)
    if visualize and dim == 2:
        import matplotlib.pyplot as plt

        # Vytvoříme mřížku a vyhodnotíme funkci
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Heatmapa: vyplněné kontury
        levels = 50
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)

        # Volitelně: přidáme contour čáry (izolinie)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Differential Evolution (DE/rand/1/bin) – Heatmap')

        # Vykreslíme cestu nejlepšího jedince
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako čára
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='DE path')
            
            # Start bod (modrý)
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue', 
                      edgecolor='white', linewidth=2, marker='o', 
                      label='Start', zorder=5)
            
            # Finální bod (zelený)
            ax.scatter([best_x[0]], [best_x[1]], s=200, c='lime', 
                      edgecolor='white', linewidth=2, marker='*', 
                      label='Best', zorder=5)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        msg = f"DE: best=({best_x[0]:.4f}, {best_x[1]:.4f}), f={best_f:.6g}, gen={max_gen}, NP={NP}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))

        plt.tight_layout()
        plt.show()

    return best_x, best_f

def teaching_learning_based_optimization(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    NP: int = 30,  # velikost populace (počet studentů ve třídě)
    max_OFE: int = 3000,  # maximální počet vyhodnocení cílové funkce
    seed: Optional[int] = None,  # náhodný seed pro reprodukovatelnost
    visualize: bool = False,  # vizualizace (jen 2D)
    num_points: int = 200,  # hustota mřížky pro heatmapu
) -> Tuple[List[float], float]:
    """
    Teaching-Learning Based Optimization (TLBO) - algoritmus inspirovaný procesem výuky ve třídě.
    
    Princip algoritmu:
      TLBO simuluje proces učení ve třídě, kde studenti (řešení) se učí od učitele (nejlepší řešení)
      a také od sebe navzájem. Algoritmus se skládá ze dvou fází:
      
      1. TEACHER PHASE (fáze učitele):
         - Učitel je nejlepší student ve třídě (nejlepší řešení v populaci)
         - Všichni studenti se učí od učitele a posouvají se směrem k němu
         - Nová pozice = stará pozice + r * (učitel - TF * průměr)
         - TF (Teaching Factor) = 1 nebo 2 (určuje intenzitu učení)
      
      2. LEARNER PHASE (fáze vzájemného učení):
         - Každý student interaguje s náhodně vybraným jiným studentem
         - Student se učí od lepšího studenta ve dvojici
         - Pokud je druhý student lepší: nová pozice = stará + r * (druhý - první)
         - Pokud je první student lepší: nová pozice = stará + r * (první - druhý)
    
    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi [(low, high), ...].
    NP : int
        Počet studentů ve třídě (velikost populace), typicky 20-50.
    max_OFE : int
        Maximální počet vyhodnocení cílové funkce (Object Function Evaluations).
        Algoritmus skončí po dosažení tohoto limitu.
    seed : Optional[int]
        Seed pro reprodukovatelnost výsledků.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D funkce).
    num_points : int
        Hustota mřížky pro heatmapu vizualizace.
    
    Návrat
    ------
    (best_position, best_fitness) : Tuple[List[float], float]
        Nejlepší nalezené řešení (nejlepší student) a jeho fitness hodnota.
    
    """
    
    # Vytvoříme generátor náhodných čísel s daným seedem (pro reprodukovatelnost)
    rng = random.Random(seed)
    
    # Počet dimenzí (např. pro 2D funkci je dim=2, pro zadání D=30)
    dim = len(bounds)
    
    # Pomocná třída pro reprezentaci studenta ve třídě
    # Každý student má pozici v prostoru řešení a fitness hodnotu (známku)
    class Student:
        def __init__(self, position: List[float]):
            self.position = position  # Aktuální pozice studenta v prostoru řešení
            self.fitness = float('inf')  # Fitness hodnota (známka) - čím nižší, tím lepší
    
    # Pomocná funkce: clamp (oříznutí) hodnoty do mezí
    # Zajišťuje, že student nezabloudí mimo definovaný prostor
    def clamp(value: float, low: float, high: float) -> float:
        """Ořízne hodnotu do intervalu [low, high]."""
        if value < low:
            return low
        elif value > high:
            return high
        else:
            return value
    
    # 1) Inicializace populace studentů (třídy)
    # Vytvoříme NP studentů s náhodnými pozicemi v daných mezích
    population: List[Student] = []
    first_student_position = None  # Uložíme první pozici pro vizualizaci startu
    
    # Počítadlo vyhodnocení cílové funkce (OFE = Objective Function Evaluations)
    # Algoritmus skončí, když dosáhneme max_OFE
    OFE_count = 0
    
    for idx in range(NP):
        # Náhodná počáteční pozice v mezích
        position = []
        for low, high in bounds:
            # Pozice v rozsahu [low, high] z rovnoměrného rozdělení
            pos = rng.uniform(low, high)
            position.append(pos)
        
        # Vytvoříme studenta s touto pozicí
        student = Student(position)
        
        # Vyhodnotíme počáteční fitness (známku studenta)
        # Pro minimalizaci: čím nižší fitness, tím lepší student
        student.fitness = objective(position)
        OFE_count += 1  # Započítáme vyhodnocení funkce
        
        # Přidáme studenta do třídy (populace)
        population.append(student)
        
        # Uložíme první pozici jako reprezentant počátečního stavu
        if idx == 0:
            first_student_position = list(position)
    
    # Najdeme učitele (nejlepšího studenta v počáteční populaci)
    # Učitel je student s nejnižší fitness hodnotou
    teacher = min(population, key=lambda s: s.fitness)
    best_position = list(teacher.position)
    best_fitness = teacher.fitness
    
    # Pro vizualizaci: ukládáme cestu učitele v čase
    path: List[Tuple[List[float], float]] = [
        (first_student_position, objective(first_student_position)),
        (list(best_position), best_fitness)
    ]
    
    # 2) Hlavní smyčka - iterace TLBO
    # Pokračujeme dokud nevyčerpáme max_OFE vyhodnocení
    # Každá iterace má dvě fáze: Teacher Phase a Learner Phase
    iteration = 0
    while OFE_count < max_OFE:
        iteration += 1
        
        # ================================================================
        # FÁZE 1: TEACHER PHASE (Učení od učitele)
        # ================================================================
        # V této fázi se všichni studenti učí od učitele (nejlepšího studenta)
        
        # Najdeme aktuálního učitele (může se měnit v každé iteraci)
        teacher = min(population, key=lambda s: s.fitness)
        
        # Vypočítáme průměrnou pozici všech studentů ve třídě
        # Mean = (X_1 + X_2 + ... + X_NP) / NP
        mean_position = [0.0] * dim
        for student in population:
            for d in range(dim):
                mean_position[d] += student.position[d]
        for d in range(dim):
            mean_position[d] /= NP
        
        # Pro každého studenta provedeme Teacher Phase
        for student in population:
            # Teaching Factor (TF) - určuje intenzitu učení
            # TF je náhodně 1 nebo 2 (s rovnou pravděpodobností)
            # TF=1 znamená silnější učení, TF=2 slabší učení
            TF = rng.choice([1, 2])
            
            # ================================================================
            # HLAVNÍ ROVNICE TEACHER PHASE:
            # ================================================================
            # X_new = X_old + r * (X_teacher - TF * X_mean)
            #
            # kde:
            #   X_old     = aktuální pozice studenta
            #   X_teacher = pozice učitele (nejlepší student)
            #   X_mean    = průměrná pozice všech studentů
            #   TF        = Teaching Factor (1 nebo 2)
            #   r         = náhodné číslo [0, 1] pro každou dimenzi
            #
            # Intuice:
            #   - (X_teacher - TF * X_mean) je rozdíl mezi učitelem a průměrem třídy
            #   - Student se posouvá směrem k tomuto rozdílu
            #   - Čím lepší učitel, tím více se studenti posouvají směrem k němu
            # ================================================================
            
            # Vytvoříme novou pozici pro studenta
            new_position = []
            for d in range(dim):
                # Náhodné číslo pro tuto dimenzi (každá dimenze má své r)
                r = rng.uniform(0, 1)
                
                # Difference_Mean je rozdíl mezi učitelem a váženým průměrem
                difference_mean = teacher.position[d] - TF * mean_position[d]
                
                # Nová souřadnice podle Teacher Phase rovnice
                new_coord = student.position[d] + r * difference_mean
                
                # Ošetření hranic - pokud student vylétne mimo prostor, vrátíme ho zpět
                low, high = bounds[d]
                new_coord = clamp(new_coord, low, high)
                
                new_position.append(new_coord)
            
            # Vyhodnotíme fitness nové pozice
            # DŮLEŽITÉ: počítáme OFE pouze pro nové vyhodnocení
            if OFE_count >= max_OFE:
                break  # Pokud jsme vyčerpali OFE, přerušíme Teacher Phase
            
            new_fitness = objective(new_position)
            OFE_count += 1
            
            # GREEDY SELECTION: Akceptujeme novou pozici pouze pokud je lepší nebo stejná
            # Tím zajistíme, že se student nikdy nezhorší
            if new_fitness <= student.fitness:
                student.position = new_position
                student.fitness = new_fitness
                
                # Pokud je tento student lepší než dosavadní učitel, aktualizujeme best
                if new_fitness < best_fitness:
                    best_position = list(new_position)
                    best_fitness = new_fitness
        
        # Pokud jsme vyčerpali OFE v Teacher Phase, ukončíme iteraci
        if OFE_count >= max_OFE:
            break
        
        # ================================================================
        # FÁZE 2: LEARNER PHASE (Vzájemné učení studentů)
        # ================================================================
        # V této fázi se studenti učí od sebe navzájem
        # Každý student interaguje s náhodně vybraným jiným studentem
        
        for i in range(NP):
            student_i = population[i]
            
            # Vybereme náhodně jiného studenta j (musí být jiný než i)
            # Používáme range místo sample pro kompatibilitu se seed
            possible_indices = list(range(NP))
            possible_indices.remove(i)  # Odstraníme index i
            j = rng.choice(possible_indices)  # Náhodně vybereme jiného studenta
            student_j = population[j]
            
            # ================================================================
            # HLAVNÍ ROVNICE LEARNER PHASE:
            # ================================================================
            # Pokud je student i lepší než student j (f_i < f_j):
            #   X_new = X_i + r * (X_i - X_j)
            # Jinak (student j je lepší):
            #   X_new = X_i + r * (X_j - X_i)
            #
            # kde:
            #   X_i, X_j  = pozice dvou náhodně vybraných studentů
            #   r         = náhodné číslo [0, 1]
            #
            # Intuice:
            #   - Student se učí od lepšího studenta ve dvojici
            #   - Pokud jsem lepší, pohybuji se pryč od horšího (diverzifikace)
            #   - Pokud jsem horší, pohybuji se směrem k lepšímu (intensifikace)
            # ================================================================
            
            # Vytvoříme novou pozici na základě vzájemného učení
            new_position = []
            for d in range(dim):
                # Náhodné číslo pro tuto dimenzi
                r = rng.uniform(0, 1)
                
                # Porovnáme fitness obou studentů a určíme směr učení
                if student_i.fitness < student_j.fitness:
                    # Student i je lepší - pohybujeme se pryč od j
                    new_coord = student_i.position[d] + r * (student_i.position[d] - student_j.position[d])
                else:
                    # Student j je lepší - pohybujeme se směrem k j
                    new_coord = student_i.position[d] + r * (student_j.position[d] - student_i.position[d])
                
                # Ošetření hranic
                low, high = bounds[d]
                new_coord = clamp(new_coord, low, high)
                
                new_position.append(new_coord)
            
            # Vyhodnotíme fitness nové pozice
            if OFE_count >= max_OFE:
                break  # Pokud jsme vyčerpali OFE, přerušíme Learner Phase
            
            new_fitness = objective(new_position)
            OFE_count += 1
            
            # GREEDY SELECTION: Akceptujeme novou pozici pouze pokud je lepší nebo stejná
            if new_fitness <= student_i.fitness:
                student_i.position = new_position
                student_i.fitness = new_fitness
                
                # Pokud je tento student lepší než dosavadní best, aktualizujeme
                if new_fitness < best_fitness:
                    best_position = list(new_position)
                    best_fitness = new_fitness
        
        # Uložíme aktuálního učitele pro vizualizaci
        # Najdeme nejlepšího studenta po této iteraci
        current_teacher = min(population, key=lambda s: s.fitness)
        if current_teacher.fitness < best_fitness:
            best_position = list(current_teacher.position)
            best_fitness = current_teacher.fitness
        path.append((list(best_position), best_fitness))
    
    # 3) Finální výsledek
    # Po vyčerpání max_OFE najdeme finálně nejlepšího studenta (učitele)
    final_teacher = min(population, key=lambda s: s.fitness)
    best_position = final_teacher.position
    best_fitness = final_teacher.fitness
    
    # 4) Vizualizace (heatmapa pro 2D)
    # Pokud je požadována vizualizace a funkce je 2D, vytvoříme heatmapu
    if visualize and dim == 2:
        import matplotlib.pyplot as plt
        
        # Vytvoříme mřížku a vyhodnotíme cílovou funkci pro heatmapu
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)
        
        # Vytvoříme figure a axes pro vykreslení
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Heatmapa: vyplněné kontury zobrazující hodnoty cílové funkce
        levels = 50  # Počet úrovní barev
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar (legendu barev) pro zobrazení hodnot funkce
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)
        
        # Volitelně: přidáme contour čáry (izolinie) pro lepší čitelnost
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        
        # Popisky os
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Teaching-Learning Based Optimization (TLBO) – Heatmap')
        
        # Vykreslíme cestu učitele (nejlepšího studenta) přes všechny iterace
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako bílá čára
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='Učitel v čase')
            
            # Zvýrazníme body, kde došlo ke zlepšení (fitness klesla)
            improvement_indices = [0]
            for i in range(1, len(path)):
                if path[i][1] < path[i-1][1]:  # Pokud je fitness lepší než předchozí
                    improvement_indices.append(i)
            
            # Vykreslíme body zlepšení žlutými kroužky
            if improvement_indices:
                imp_x = [path_x[i] for i in improvement_indices]
                imp_y = [path_y[i] for i in improvement_indices]
                ax.scatter(imp_x, imp_y, s=80, c='yellow', edgecolor='white',
                          linewidth=1.5, marker='o', zorder=6,
                          label=f'Zlepšení ({len(improvement_indices)}x)')
            
            # Start bod (modrý kruh) - první student v populaci
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue',
                      edgecolor='white', linewidth=2, marker='o',
                      label='Start', zorder=5)
            
            # Finální nejlepší bod (zelená hvězda)
            ax.scatter([best_position[0]], [best_position[1]], s=200, c='lime',
                      edgecolor='white', linewidth=2, marker='*',
                      label='Best', zorder=5)
        
        # Legenda s popisem
        ax.legend(loc='upper right')
        
        # Mřížka pro lepší orientaci
        ax.grid(True, alpha=0.3)
        
        # Textový box s informacemi o výsledku
        msg = f"TLBO: best=({best_position[0]:.4f}, {best_position[1]:.4f}), f={best_fitness:.6g}, iter={iteration}, NP={NP}, OFE={OFE_count}/{max_OFE}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))
        
        # Úprava rozložení a zobrazení grafu
        plt.tight_layout()
        plt.show()
    
    return best_position, best_fitness