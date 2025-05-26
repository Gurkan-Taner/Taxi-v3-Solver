from TaxiSolver import TaxiSolver


def asking_load_model() -> str:
    model_path = ""
    load_model = str(input("Importer un model pour le DQN ? [y/N] "))
    if load_model == "y":
        model_path = str(input("Entrer le chemin du model : [./taxi_dqn_model.pth] "))
        return model_path if model_path != "" else "./taxi_dqn_model.pth"


def load_model(solver: TaxiSolver, model_path: str):
    solver.dqn.load_model(model_path)


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
        model_path = asking_load_model()
        if model_path:
            load_model(solver, model_path)

        time_limit = int(input("Temps limite en secondes : "))
        solver.run_time_mode(time_limit)
    elif mode == 3:
        model_path = asking_load_model()
        if model_path:
            load_model(solver, model_path)

        solver.run_time_mode()

    solver.env.close()


if __name__ == "__main__":
    main()
