import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('tp2_atdn_donnees.csv', encoding='utf-8')
data.columns = ['Humidite', 'Temperature', 'pH_sol', 'Precipitations', 'Type_de_sol', 'Rendement']
print(data.head())


le = LabelEncoder()
data['Type_de_sol'] = le.fit_transform(data['Type_de_sol'])

# Définition de la fonction objectif pour l'optimisation bayésienne
def objective(params):
    temp, hum = params
    return -(3 * np.sin(temp / 10) + 2 * np.cos(hum / 20))  # minimise la fonction négative

# Définition de l'espace de recherche
space = [Real(10, 35, name='temp'), Real(30, 80, name='hum')]

# Optimisation bayésienne des conditions
res = gp_minimize(objective, space, n_calls=20, random_state=42)


plt.figure(figsize=(10, 5))
plot_convergence(res)
plt.show()


print("Meilleures conditions pour maximiser le rendement :")
print(f"Température: {res.x[0]:.2f}°C, Humidité: {res.x[1]:.2f}%")

# Optimisation hyperparamètres  Random Forest
def evaluate_model(params):
    n_estimators, max_depth = params
    model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    model.fit(data[['Temperature', 'Humidite']], data['Rendement'])
    return -model.score(data[['Temperature', 'Humidite']], data['Rendement'])  # Minimiser l'erreur

space_rf = [Real(10, 200, name='n_estimators'), Real(3, 20, name='max_depth')]

res_rf = gp_minimize(evaluate_model, space_rf, n_calls=20, random_state=42)

print("Meilleurs hyperparamètres pour Random Forest :")
print(f"n_estimators: {int(res_rf.x[0])}, max_depth: {int(res_rf.x[1])}")
