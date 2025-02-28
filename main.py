from TaxiSolver import TaxiSolver


def main():
    print("=== PROJET TAXI-V3 ===")
    solver = TaxiSolver()

    print("\nChoisissez un mode :")
    print("1. Mode utilisateur (configuration manuelle)")
    print("2. Mode temps limité (paramètres optimisés)")
    print("3. Mode temps illimité (paramètres optimisés)")

    mode = int(input("Votre choix (1-3) : "))

    if mode == 1:
        solver.run_user_mode()
    elif mode == 2:
        time_limit = int(input("Temps limite en secondes : "))
        solver.run_time_mode(time_limit)
    elif mode == 3:
        solver.run_time_mode()


if __name__ == "__main__":
    main()
