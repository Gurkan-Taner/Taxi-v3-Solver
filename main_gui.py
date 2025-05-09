import streamlit as st
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import gymnasium as gym
import numpy as np
from TaxiSolver import TaxiSolver
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.title("Projet Taxi-v3 Reinforcement Learning")
    st.sidebar.header("Configuration")

    # Création de l'environnement et de l'agent
    @st.cache_resource
    def create_solver():
        return TaxiSolver()

    solver = create_solver()

    mode = st.sidebar.radio(
        "Choisissez un mode",
        ["Mode utilisateur", "Mode temps limité", "Mode temps illimité"],
    )

    st.sidebar.subheader("Paramètres communs")
    training_episodes = st.sidebar.slider(
        "Nombre d'épisodes d'entraînement", 100, 10000, 1000, step=5
    )
    testing_episodes = st.sidebar.slider(
        "Nombre d'épisodes de test", 10, 1000, 100, step=5
    )

    load_model = st.sidebar.checkbox("Charger un modèle pré-entraîné")
    model_path = None
    if load_model:
        model_path = st.sidebar.text_input("Chemin du modèle", "./taxi_dqn_model.pth")

    def plot_results(q_rewards=None, dqn_rewards=None, q_steps=None, dqn_steps=None):
        if q_rewards is not None and dqn_rewards is not None:
            st.subheader("Récompenses obtenues")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=q_rewards, mode="lines", name="Q-Learning"))
            fig1.add_trace(go.Scatter(y=dqn_rewards, mode="lines", name="DQN"))
            fig1.update_layout(
                title="Récompenses par épisode",
                xaxis_title="Épisode",
                yaxis_title="Récompense totale",
            )
            st.plotly_chart(fig1, use_container_width=True)

            if len(q_rewards) >= 100 and len(dqn_rewards) >= 100:
                q_avg = np.mean(q_rewards[-100:])
                dqn_avg = np.mean(dqn_rewards[-100:])

                cols = st.columns(2)
                cols[0].metric(
                    "Récompense moyenne Q-Learning (100 derniers)", f"{q_avg:.2f}"
                )
                cols[1].metric(
                    "Récompense moyenne DQN (100 derniers)", f"{dqn_avg:.2f}"
                )

        if q_steps is not None and dqn_steps is not None:
            st.subheader("Nombre d'étapes par épisode")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=q_steps, mode="lines", name="Q-Learning"))
            fig2.add_trace(go.Scatter(y=dqn_steps, mode="lines", name="DQN"))
            fig2.update_layout(
                title="Étapes par épisode",
                xaxis_title="Épisode",
                yaxis_title="Nombre d'étapes",
            )
            st.plotly_chart(fig2, use_container_width=True)

            if len(q_steps) >= 100 and len(dqn_steps) >= 100:
                q_steps_avg = np.mean(q_steps[-100:])
                dqn_steps_avg = np.mean(dqn_steps[-100:])

                cols = st.columns(2)
                cols[0].metric(
                    "Étapes moyennes Q-Learning (100 derniers)", f"{q_steps_avg:.2f}"
                )
                cols[1].metric(
                    "Étapes moyennes DQN (100 derniers)", f"{dqn_steps_avg:.2f}"
                )

    def train_and_test():
        with st.spinner("Entraînement des modèles en cours..."):
            if mode == "Mode utilisateur":
                alpha = st.session_state.get("alpha", 0.6)
                gamma = st.session_state.get("gamma", 0.7)
                epsilon = st.session_state.get("epsilon", 0.8)
                min_epsilon = st.session_state.get("min_epsilon", 0.1)
                epsilon_decay = st.session_state.get("epsilon_decay", 1e-4)
            else:
                alpha = 0.6
                gamma = 0.7
                epsilon = 0.8
                min_epsilon = 0.1
                epsilon_decay = 1e-4

            q_train_rewards, q_train_steps = solver.q_learning.train_q_learning(
                training_episodes,
                alpha,
                gamma,
                epsilon,
                min_epsilon,
                epsilon_decay,
            )

            if model_path:
                solver.dqn.load_model(model_path)
                dqn_train_rewards, dqn_train_steps = [], []
                st.success(f"Modèle DQN chargé depuis {model_path}")
            else:
                dqn_train_rewards, dqn_train_steps = solver.dqn.train_dqn(
                    training_episodes
                )

            st.success("Entraînement terminé!")

            st.subheader("Résultats d'entraînement")
            plot_results(
                q_train_rewards, dqn_train_rewards, q_train_steps, dqn_train_steps
            )

        with st.spinner("Test des modèles en cours..."):
            env = gym.make("Taxi-v3")

            time_limit = None
            if mode == "Mode temps limité":
                time_limit = st.session_state.get("time_limit", 60)

            start_time = time()
            q_test_rewards, q_test_steps = solver.q_learning.test_q_learning(
                env=env,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )

            dqn_test_rewards, dqn_test_steps = solver.dqn.test_dqn(
                env=env,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )

            env.close()

            st.success("Tests terminés!")

            st.subheader("Résultats de test")
            plot_results(q_test_rewards, dqn_test_rewards, q_test_steps, dqn_test_steps)

            q_avg_reward = np.mean(q_test_rewards)
            dqn_avg_reward = np.mean(dqn_test_rewards)
            q_avg_steps = np.mean(q_test_steps)
            dqn_avg_steps = np.mean(dqn_test_steps)

            comparison_df = pd.DataFrame(
                {
                    "Algorithme": ["Q-Learning", "DQN"],
                    "Récompense moyenne": [q_avg_reward, dqn_avg_reward],
                    "Étapes moyennes": [q_avg_steps, dqn_avg_steps],
                }
            )

            st.subheader("Comparaison des performances")
            st.dataframe(comparison_df, use_container_width=True)

            fig = px.bar(
                comparison_df,
                x="Algorithme",
                y=["Récompense moyenne", "Étapes moyennes"],
                barmode="group",
                title="Comparaison Q-Learning vs DQN",
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Sauvegarder le modèle DQN"):
                save_path = "./taxi_dqn_model.pth"
                solver.dqn.save_model(save_path)
                st.success(f"Modèle sauvegardé sous {save_path}")

    if mode == "Mode utilisateur":
        st.sidebar.subheader("Paramètres Q-Learning")
        st.sidebar.slider(
            "Taux d'apprentissage (alpha)", 0.1, 1.0, 0.6, 0.1, key="alpha"
        )
        st.sidebar.slider(
            "Facteur d'actualisation (gamma)", 0.1, 0.99, 0.7, 0.05, key="gamma"
        )
        st.sidebar.slider("Epsilon initial", 0.1, 1.0, 0.8, 0.1, key="epsilon")
        st.sidebar.slider("Epsilon minimal", 0.01, 0.5, 0.1, 0.01, key="min_epsilon")
        st.sidebar.slider(
            "Taux de décroissance d'epsilon",
            0.0001,
            0.1,
            0.0001,
            0.0001,
            format="%.4f",
            key="epsilon_decay",
        )

        algo_choice = st.sidebar.multiselect(
            "Algorithmes à exécuter",
            ["Q-Learning", "DQN"],
            default=["Q-Learning", "DQN"],
        )

        st.write(
            """
        ## Mode utilisateur
        Ce mode vous permet de configurer manuellement les paramètres des algorithmes 
        d'apprentissage par renforcement pour résoudre l'environnement Taxi-v3.
        """
        )

    elif mode == "Mode temps limité":
        st.sidebar.subheader("Configuration du temps")
        st.sidebar.slider("Temps limite (secondes)", 10, 300, 60, key="time_limit")

        st.write(
            f"""
        ## Mode temps limité
        Les algorithmes s'exécuteront avec un temps limité de {st.session_state.get('time_limit', 60)} secondes.
        Les paramètres sont optimisés automatiquement.
        """
        )

    else:
        st.write(
            """
        ## Mode temps illimité
        Les algorithmes s'exécuteront sans limite de temps.
        Les paramètres sont optimisés automatiquement.
        """
        )

    if st.button("Lancer l'apprentissage et les tests"):
        train_and_test()

    with st.expander("À propos de l'environnement Taxi-v3"):
        st.write(
            """
        L'environnement Taxi-v3 est un problème classique de reinforcement learning.
        
        **Description :**
        - Un taxi se déplace sur une grille 5x5
        - Il doit prendre un passager à un emplacement et le déposer à un autre
        - Il y a 4 emplacements possibles (R, G, Y, B)
        - Le taxi peut se déplacer (Nord, Sud, Est, Ouest), prendre ou déposer un passager
        
        **Récompenses :**
        - -1 par action (pénalité pour chaque étape)
        - -10 pour un ramassage ou dépôt illégal
        - +20 pour une livraison réussie
        
        **Objectif :** Maximiser la récompense totale en minimisant le nombre d'étapes nécessaires.
        """
        )

        st.image(
            "https://www.gymlibrary.dev/_images/taxi.gif",
            caption="Illustration de l'environnement Taxi-v3",
        )


if __name__ == "__main__":
    main()
