# Taxi-v3 solver

Small project in python to solve taxi-v3 environment provided by gym.

## Architecture of project

├── algos
│ ├── DQLearning
│ └── QLearning

This folder have our algorithms script and classes

├── main_console.py
├── main_gui.py

Our main programs to run

├── requirements.txt

Dependencies file

└── test
├── game_test.py
└── old_maingui.py

Test folder to begin to learn the gym environment

## How to run

First, create virtual env and install dependencies :

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Then, run streamlit server :

```console
$ streamlit run main_gui.py
```

Then, in the launched window, choose your mod and your hyperparameters in the sidebar to train and test Q-learning and DQN. You will see many charts and data to vizualise.
