import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import seaborn as sns

# Cargar los datos
df1 = pd.read_excel('requerimiento1.xlsx')
df2 = pd.read_excel('requerimiento2.xlsx')
df3 = pd.read_excel('requerimiento3.xlsx')

# Requerimiento 1: Regresión Lineal Simple
X1 = df1[['pacientes atendidos']]
y1 = df1['tiempo de espera (min)']
modelo1 = LinearRegression().fit(X1, y1)

# Requerimiento 2: Regresión Lineal Múltiple
X2 = df2[['tiempo de espera (min)', 'duracion de la consulta (min)', 'personal medico disponible']]
y2 = df2['satisfaccion general']
modelo2 = LinearRegression().fit(X2, y2)

# Requerimiento 3: Regresión Lineal Múltiple
X3 = df3[['tiempo de espera (min)', 'satisfaccion general (1-10)']]
y3 = df3['probabilidad de recomendacion (%)']
modelo3 = LinearRegression().fit(X3, y3)

# Funciones de predicción
def predecir_tiempo_espera(pacientes):
    return modelo1.predict([[pacientes]])[0]

def predecir_satisfaccion(tiempo_espera, duracion_consulta, personal_medico):
    return modelo2.predict([[tiempo_espera, duracion_consulta, personal_medico]])[0]

def predecir_recomendacion(tiempo_espera, satisfaccion):
    return modelo3.predict([[tiempo_espera, satisfaccion]])[0]

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Predictor Mundosalud")
root.geometry("800x600")
root.configure(bg='#f0f0f0')

style = ttk.Style()
style.theme_use('clam')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
style.configure('TEntry', font=('Arial', 10))
style.configure('TButton', font=('Arial', 10, 'bold'))

main_frame = ttk.Frame(root, padding="20", style='TFrame')
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
main_frame.configure(style='TFrame')

# Requerimiento 1
ttk.Label(main_frame, text="Requerimiento 1: Predicción de Tiempo de Espera", font=('Arial', 12, 'bold')).grid(column=0, row=0, columnspan=2, pady=10)
ttk.Label(main_frame, text="Pacientes Atendidos:").grid(column=0, row=1, sticky=tk.W)
pacientes_entry = ttk.Entry(main_frame, width=30)
pacientes_entry.grid(column=1, row=1)
tiempo_espera_label = ttk.Label(main_frame, text="")
tiempo_espera_label.grid(column=0, row=2, columnspan=2)

# Requerimiento 2
ttk.Label(main_frame, text="Requerimiento 2: Predicción de Satisfacción", font=('Arial', 12, 'bold')).grid(column=0, row=3, columnspan=2, pady=10)
ttk.Label(main_frame, text="Duración de la Consulta (min):").grid(column=0, row=4, sticky=tk.W)
duracion_entry = ttk.Entry(main_frame, width=30)
duracion_entry.grid(column=1, row=4)
ttk.Label(main_frame, text="Personal Médico Disponible:").grid(column=0, row=5, sticky=tk.W)
personal_entry = ttk.Entry(main_frame, width=30)
personal_entry.grid(column=1, row=5)
satisfaccion_label = ttk.Label(main_frame, text="")
satisfaccion_label.grid(column=0, row=6, columnspan=2)

# Requerimiento 3
ttk.Label(main_frame, text="Requerimiento 3: Predicción de Probabilidad de Recomendación", font=('Arial', 12, 'bold')).grid(column=0, row=7, columnspan=2, pady=10)
recomendacion_label = ttk.Label(main_frame, text="")
recomendacion_label.grid(column=0, row=8, columnspan=2)

# Función para actualizar gráficos
def actualizar_graficos(pacientes, tiempo_espera, satisfaccion, probabilidad):
    fig.clear()
    
    # Gráfico de dispersión para Requerimiento 1
    ax1 = fig.add_subplot(121)
    sns.scatterplot(x='pacientes atendidos', y='tiempo de espera (min)', data=df1, ax=ax1)
    ax1.plot(df1['pacientes atendidos'], modelo1.predict(df1[['pacientes atendidos']]), color='red')
    ax1.scatter([pacientes], [tiempo_espera], color='green', s=100, marker='*')
    ax1.set_xlabel('Pacientes Atendidos')
    ax1.set_ylabel('Tiempo de Espera (min)')
    ax1.set_title('Predicción de Tiempo de Espera')
    
    # Gráfico de barras para Requerimiento 2 y 3
    ax2 = fig.add_subplot(122)
    sns.barplot(x=['Satisfacción', 'Prob. Recomendación'], y=[satisfaccion, probabilidad], ax=ax2)
    ax2.set_ylim(0, 100)
    ax2.set_title('Satisfacción y Probabilidad de Recomendación')
    
    canvas.draw()

# Función para hacer predicciones
def hacer_predicciones():
    try:
        pacientes = float(pacientes_entry.get())
        tiempo_espera = predecir_tiempo_espera(pacientes)
        tiempo_espera_label.config(text=f"Tiempo de Espera Predicho: {tiempo_espera:.2f} min")
        
        duracion = float(duracion_entry.get())
        personal = float(personal_entry.get())
        satisfaccion = predecir_satisfaccion(tiempo_espera, duracion, personal)
        satisfaccion_label.config(text=f"Satisfacción Predicha: {satisfaccion:.2f}")
        
        probabilidad = predecir_recomendacion(tiempo_espera, satisfaccion)
        recomendacion_label.config(text=f"Probabilidad de Recomendación: {probabilidad:.2f}%")
        
        actualizar_graficos(pacientes, tiempo_espera, satisfaccion, probabilidad)
    except ValueError:
        tk.messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos.")

# Botón de predicción
ttk.Button(main_frame, text="Predecir", command=hacer_predicciones, style='TButton').grid(column=0, row=9, columnspan=2, pady=10)

# Crear gráficos
fig = plt.Figure(figsize=(10, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0, pady=10)

root.mainloop()