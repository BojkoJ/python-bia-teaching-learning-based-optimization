"""
Test pro Teaching-Learning Based Optimization (TLBO) - algoritmus inspirovaný procesem výuky ve třídě s vizualizací.
"""
import sys
import os

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('TkAgg')  # Explicitní backend pro zobrazení oken

from main import teaching_learning_based_optimization, ackley, sphere, rastrigin, rosenbrock, griewank, levy, get_default_bounds


def test_tlbo_sphere_with_visualization():
    """
    Spustí TLBO na Sphere funkci s heatmap vizualizací.
    """
    print("=" * 60)
    print("TLBO - Sphere funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=sphere,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_tlbo_rastrigin_with_viz():
    """
    TLBO na Rastrigin s vizualizací.
    """
    print("=" * 60)
    print("TLBO - Rastrigin funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=rastrigin,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=999,
        visualize=True,
        num_points=180,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_tlbo_ackley_with_viz():
    """
    TLBO na Ackley s vizualizací.
    """
    print("=" * 60)
    print("TLBO - Ackley funkce (2D)")
    print("=" * 60)
    
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=ackley,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_tlbo_rosenbrock_with_viz():
    """
    TLBO na Rosenbrock s vizualizací.
    """
    print("=" * 60)
    print("TLBO - Rosenbrock funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.0, 10.0), (-5.0, 10.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=rosenbrock,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=555,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (1, 1), f = 0")
    print()


def test_tlbo_griewank_with_viz():
    """
    TLBO na Griewank s vizualizací.
    """
    print("=" * 60)
    print("TLBO - Griewank funkce (2D)")
    print("=" * 60)
    
    bounds = [(-6.0, 6.0), (-6.0, 6.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=griewank,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=777,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_tlbo_levy_with_viz():
    """
    TLBO na Levy s vizualizací.
    """
    print("=" * 60)
    print("TLBO - Levy funkce (2D)")
    print("=" * 60)
    
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=levy,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=333,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (1, 1), f = 0")
    print()


def test_tlbo_sphere_quick():
    """
    Rychlý test TLBO na Sphere bez vizualizace.
    """
    print("=" * 60)
    print("TLBO - Sphere funkce (rychlý test)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = teaching_learning_based_optimization(
        objective=sphere,
        bounds=bounds,
        NP=30,
        max_OFE=3000,
        seed=123,
        visualize=False,
    )
    
    print(f"Výsledek: x = {best_x}, f(x) = {best_f:.8f}")
    
    # Kontrola: pro Sphere by mělo být blízko nuly
    assert best_f < 0.1, f"TLBO na Sphere selhalo: f = {best_f} > 0.1"
    print("✓ Test prošel!")
    print()


def test_tlbo_parameter_comparison():
    """
    Porovnání TLBO s různými velikostmi populace na Sphere funkci.
    """
    print("=" * 60)
    print("TLBO - Porovnání velikostí populace")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    configs = [
        {"NP": 10, "max_OFE": 3000, "label": "Malá populace (NP=10)"},
        {"NP": 20, "max_OFE": 3000, "label": "Střední populace (NP=20)"},
        {"NP": 30, "max_OFE": 3000, "label": "Velká populace (NP=30)"},
        {"NP": 50, "max_OFE": 3000, "label": "Velmi velká populace (NP=50)"},
    ]
    
    for i, config in enumerate(configs, 1):
        label = config.pop("label")
        print(f"\nKonfigurace {i} ({label}):")
        print(f"  NP={config['NP']}, max_OFE={config['max_OFE']}")
        
        best_x, best_f = teaching_learning_based_optimization(
            objective=sphere,
            bounds=bounds,
            seed=42,
            visualize=False,
            **config
        )
        
        print(f"  Výsledek: x = [{best_x[0]:.6f}, {best_x[1]:.6f}], f(x) = {best_f:.8f}")
    
    print("\n✓ Porovnání dokončeno!")
    print()




if __name__ == "__main__":
    # Hlavní vizualizace na Rastrigin (složitější funkce s mnoha lokálními minimy)
    test_tlbo_rastrigin_with_viz()
    
    # Rychlý test bez vizualizace
    test_tlbo_sphere_quick()
    
    # Další vizualizace
    # test_tlbo_sphere_with_visualization()
    # test_tlbo_ackley_with_viz()
    # test_tlbo_rosenbrock_with_viz()
    # test_tlbo_griewank_with_viz()
    # test_tlbo_levy_with_viz()
    
    # Porovnání různých konfigurací parametrů
    test_tlbo_parameter_comparison()
    
    print("=" * 60)
    print("VŠECHNY TESTY DOKONČENY")
    print("=" * 60)
