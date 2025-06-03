import streamlit as st
from time import time
import pandas as pd
import gymnasium as gym
import numpy as np
from TaxiSolver import TaxiSolver
import plotly.express as px
import plotly.graph_objects as go


class TaxiApp:
    def __init__(self):
        self.solver = self._create_solver()

    @staticmethod
    @st.cache_resource
    def _create_solver():
        return TaxiSolver()

    def setup_page(self):
        st.title("Projet Taxi-v3 Reinforcement Learning")
        st.sidebar.header("Configuration")

        if "can_save_dqn" not in st.session_state:
            st.session_state["can_save_dqn"] = False

    def get_sidebar_config(self):
        config = {}

        config["mode"] = st.sidebar.radio(
            "Choisissez un mode",
            ["Mode utilisateur", "Mode temps limité", "Mode temps illimité"],
        )

        st.sidebar.subheader("Paramètres communs")
        config["training_episodes"] = st.sidebar.slider(
            "Nombre d'épisodes d'entraînement", 100, 10000, 1000, step=5
        )
        config["testing_episodes"] = st.sidebar.slider(
            "Nombre d'épisodes de test", 10, 1000, 100, step=5
        )

        config["load_model"] = st.sidebar.checkbox("Charger un modèle pré-entraîné")
        config["model_path"] = None

        if config["load_model"]:
            config["model_path"] = st.sidebar.text_input(
                "Chemin du modèle", "./taxi_dqn_model.pth"
            )

        config["show_visualization"] = st.sidebar.checkbox(
            "Afficher la visualisation en direct", value=False
        )
        if config["show_visualization"]:
            config["visualization_speed"] = st.sidebar.slider(
                "Vitesse de la visualisation", 0.01, 1.0, 0.1, 0.01
            )
            config["visualization_episodes"] = st.sidebar.slider(
                "Nombre d'épisodes à visualiser", 1, 10, 3, step=1
            )
        else:
            config["visualization_speed"] = 0
            config["visualization_episodes"] = 0

        return config

    def get_mode_specific_config(self, mode):
        config = {}

        if mode == "Mode utilisateur":
            config.update(self._setup_user_mode())
        elif mode == "Mode temps limité":
            config.update(self._setup_time_limited_mode())
        else:
            config.update(self._setup_unlimited_mode())

        return config

    def _setup_user_mode(self):
        st.sidebar.subheader("Paramètres Q-Learning")

        config = {
            "alpha": st.sidebar.slider(
                "Taux d'apprentissage (alpha)", 0.1, 1.0, 0.6, 0.1, key="alpha"
            ),
            "gamma": st.sidebar.slider(
                "Facteur de réduction (gamma)", 0.1, 0.99, 0.7, 0.05, key="gamma"
            ),
            "epsilon": st.sidebar.slider(
                "Epsilon initial", 0.1, 1.0, 0.8, 0.1, key="epsilon"
            ),
            "min_epsilon": st.sidebar.slider(
                "Epsilon minimal", 0.01, 0.5, 0.1, 0.01, key="min_epsilon"
            ),
            "epsilon_decay": st.sidebar.slider(
                "Taux de décroissance d'epsilon",
                0.0001,
                0.1,
                0.0001,
                0.0001,
                format="%.4f",
                key="epsilon_decay",
            ),
            "algo_choice": st.sidebar.multiselect(
                "Algorithmes à exécuter",
                ["Q-Learning", "DQN"],
                default=["Q-Learning", "DQN"],
            ),
        }

        st.write(
            """
        ## Mode utilisateur
        Ce mode vous permet de configurer manuellement les paramètres des algorithmes 
        d'apprentissage par renforcement pour résoudre l'environnement Taxi-v3.
        """
        )

        return config

    def _setup_time_limited_mode(self):
        st.sidebar.subheader("Configuration du temps")
        time_limit = st.sidebar.slider(
            "Temps limite (secondes)", 1, 300, 60, key="time_limit"
        )

        st.write(
            f"""
        ## Mode temps limité
        Les algorithmes s'exécuteront avec un temps limité de {time_limit} secondes.
        Les paramètres sont optimisés automatiquement.
        """
        )

        return {
            "time_limit": time_limit,
            "alpha": 0.37,
            "gamma": 0.907,
            "epsilon": 0.388,
            "min_epsilon": 0.01,
            "epsilon_decay": 0.9964,
        }

    def _setup_unlimited_mode(self):
        st.write(
            """
        ## Mode temps illimité
        Les algorithmes s'exécuteront sans limite de temps.
        Les paramètres sont optimisés automatiquement.
        """
        )

        return {
            "alpha": 0.37,
            "gamma": 0.907,
            "epsilon": 0.388,
            "min_epsilon": 0.01,
            "epsilon_decay": 0.9964,
        }

    def load_pretrained_model(self, model_path):
        if not model_path:
            return True

        try:
            self.solver.dqn.load_model(model_path)
            st.success(f"Modèle DQN chargé depuis {model_path}")
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle DQN: {e}")
            return False

    def train_models(self, training_episodes, config):
        with st.spinner("Entraînement des modèles en cours..."):
            q_train_rewards, q_train_steps = self.solver.q_learning.train_q_learning(
                training_episodes,
                config["alpha"],
                config["gamma"],
                config["epsilon"],
                config["min_epsilon"],
                config["epsilon_decay"],
            )
            dqn_train_rewards, dqn_train_steps = self.solver.dqn.train_dqn(
                training_episodes
            )

            st.success("Entraînement terminé!")

        return q_train_rewards, q_train_steps, dqn_train_rewards, dqn_train_steps

    def test_models(self, testing_episodes, mode_config):
        with st.spinner("Test des modèles en cours..."):
            env = gym.make("Taxi-v3")

            time_limit = mode_config.get("time_limit")
            start_time = time()

            q_test_rewards, q_test_steps = self.solver.q_learning.test_q_learning(
                env=env,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )

            dqn_test_rewards, dqn_test_steps = self.solver.dqn.test_dqn(
                env=env,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )

            env.close()
            st.success("Tests terminés!")

        return q_test_rewards, q_test_steps, dqn_test_rewards, dqn_test_steps

    def plot_training_results(self, q_rewards, dqn_rewards, q_steps, dqn_steps):
        st.subheader("Résultats d'entraînement")
        self._plot_results(q_rewards, dqn_rewards, q_steps, dqn_steps)

    def plot_test_results(self, q_rewards, dqn_rewards, q_steps, dqn_steps):
        st.subheader("Résultats de test")
        self._plot_results(q_rewards, dqn_rewards, q_steps, dqn_steps)

        q_avg_reward = round(np.mean(q_rewards), 2)
        dqn_avg_reward = round(np.mean(dqn_rewards), 2)
        q_avg_steps = round(np.mean(q_steps), 2)
        dqn_avg_steps = round(np.mean(dqn_steps), 2)

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

    def _plot_results(self, q_rewards, dqn_rewards, q_steps, dqn_steps):
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

            q_avg = round(np.mean(q_rewards), 2)
            dqn_avg = round(np.mean(dqn_rewards), 2)

            cols = st.columns(2)
            cols[0].metric("Récompense moyenne Q-Learning", f"{q_avg:.2f}")
            cols[1].metric("Récompense moyenne DQN", f"{dqn_avg:.2f}")

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

            q_steps_avg = np.mean(q_steps)
            dqn_steps_avg = np.mean(dqn_steps)
            cols = st.columns(2)
            cols[0].metric("Étapes moyennes Q-Learning", f"{q_steps_avg:.2f}")
            cols[1].metric("Étapes moyennes DQN", f"{dqn_steps_avg:.2f}")

    def handle_model_saving(self):
        if st.session_state.get("can_save_dqn", False):
            path_to_save = st.text_input(
                "Chemin de sauvegarde du modèle DQN", "./taxi_dqn_model.pth"
            )
            if st.button("Sauvegarder le modèle DQN", key="save_dqn_btn"):
                try:
                    self.solver.dqn.save_model(path_to_save)
                    st.success(f"Modèle sauvegardé sous {path_to_save}")

                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde: {e}")

    def show_environment_info(self):
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

    def run_training_and_testing(self, config, mode_config):
        if not self.load_pretrained_model(config.get("model_path")):
            return

        q_train_rewards, q_train_steps, dqn_train_rewards, dqn_train_steps = (
            self.train_models(config["training_episodes"], mode_config)
        )

        self.plot_training_results(
            q_train_rewards, dqn_train_rewards, q_train_steps, dqn_train_steps
        )

        q_test_rewards, q_test_steps, dqn_test_rewards, dqn_test_steps = (
            self.test_models(config["testing_episodes"], mode_config)
        )

        self.plot_test_results(
            q_test_rewards, dqn_test_rewards, q_test_steps, dqn_test_steps
        )

        st.session_state["can_save_dqn"] = True

    def run(self):
        self.setup_page()

        config = self.get_sidebar_config()

        mode_config = self.get_mode_specific_config(config["mode"])

        if st.button("Lancer l'apprentissage et les tests"):
            self.run_training_and_testing(config, mode_config)

        self.handle_model_saving()

        self.show_environment_info()


def main():
    app = TaxiApp()
    app.run()


if __name__ == "__main__":
    main()
