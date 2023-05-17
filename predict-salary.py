# se hará una prediccion de sueldo depenfiendo el puesto del empleado 
# haciendo uso de la regresion polinomica #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# leer el csv
df = pd.read_csv("Position_Salaries.csv")
print(df)

# variable independiente
x = df.iloc[:, 1:-1].values

# variable dependiente
y = df.iloc[:, -1:].values

plt.scatter(x, y)
plt.show()

# ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(x,y)

# ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)  #regresion polinomica de grado 5
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# visualizacion de los resultados del modelo lineal
plt.scatter(x,y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Modelo de regresion lineal")
plt.xlabel("Posición de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# visualizacion de los resultados del modelo polinomial
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color="red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Modelo de regresion polinomica")
plt.xlabel("Posición de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

nivel =float(input("Ingrese su nivel: "))
sueldo_predicho = lin_reg_2.predict(poly_reg.fit_transform([[nivel]]))
print(f"el sueldo que se merece es ${float(sueldo_predicho)}")