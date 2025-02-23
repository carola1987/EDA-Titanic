# importa librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# carga el dataset
df = pd.read_csv('data/train.csv')

# exploracion inicial
print(df.head())
print(df.info())
print(df.describe())

# verifica los nombres de las columnas
print("Columnas del DataFrame:", df.columns)

# Convierte nombres de columnas a minúsculas
df.columns = df.columns.str.lower().str.strip()
print(df.columns.tolist())  # verifica que las columnas ahora sean minúsculas

# Llena valores nulos en 'age'
df['age'] = df['age'].fillna(df['age'].median())

# limpia los datos
df['age'] = df['age'].fillna(df['age'].median())
df.dropna(subset=['embarked'], inplace=True)

# Filtra solo las columnas numéricas para la correlación
df_numeric = df.select_dtypes(include=['number'])

# análisis univariado = distribución de edades
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Distribución de edades de los pasajeros")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# análisis bivariado = supervivencia según clase
plt.figure(figsize=(10, 6))
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Supervivencia según clase de pasajero")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.legend(["No Sobrevivió", "Sobrevivió"])
plt.show()

# correlaciones
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de correlación del Titanic")
plt.show()
