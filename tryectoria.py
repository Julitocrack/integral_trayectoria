# PROYECTO 1: CÁLCULO PARA INTELIGENCIA ARTIFICIAL
# NOMBRE: Trayectoria
# AUTOR: Julio Rangel

# ----------------------------------------------------------------------
# -- SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS --
# ----------------------------------------------------------------------

import numpy as np               # Importa NumPy para cálculos numéricos eficientes (vectores, matrices).
import matplotlib.pyplot as plt  # Importa Matplotlib para crear figuras y ejes de la gráfica.
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D, módulo esencial para la visualización 3D.
import matplotlib.animation as animation # Importa el módulo para generar la animación de la partícula.

# ----------------------------------------------------------------------
# -- SECCIÓN 1: MÉTODO MATEMÁTICO Y CÁLCULO DE TRAYECTORIA --
# ----------------------------------------------------------------------

def f(x, y):
    """Define el campo escalar f(x,y)."""
    # Función del campo escalar: f(x,y) = x²cos(y) - y²sin(x)
    return x**2 * np.cos(y) - y**2 * np.sin(x)

def grad_f(x, y):
    """Calcula el gradiente de f(x,y) en un punto (x,y)."""
    # Derivada parcial respecto a x: df/dx = 2xcos(y) - y²cos(x)
    df_dx = 2*x*np.cos(y) - y**2*np.cos(x)
    # Derivada parcial respecto a y: df/dy = -x²sin(y) - 2ysin(x)
    df_dy = -x**2*np.sin(y) - 2*y*np.sin(x)
    # Retorna el vector gradiente [df/dx, df/dy]
    return np.array([df_dx, df_dy])

# Parámetros de la simulación
p_inicial = np.array([0.0, 0.0])  # Punto inicial de prueba (0.0, 0.0).
h = 0.05                        # Tamaño del paso (learning rate), reducido para mayor estabilidad.
limite = 5.0                      # Límite absoluto de la región Q (de -5.0 a 5.0).

# --- LÓGICA PARA EVITAR QUEDARSE EN UN PUNTO CRÍTICO (El "empujón" FUERTE) ---

# 1. Calcula la magnitud (norma) del vector gradiente en el punto inicial.
grad_inicial_norm = np.linalg.norm(grad_f(p_inicial[0], p_inicial[1]))
umbral_critico = 1e-4             # Umbral: si el gradiente es menor a 0.0001, se considera crítico.

# 2. Comprueba si el punto es (o está cerca de) un punto crítico.
if grad_inicial_norm < umbral_critico:
    print(f"¡ADVERTENCIA! Punto inicial cerca de un punto crítico (Gradiente: {grad_inicial_norm:.6f}).")
    
    # 3. Aplica una perturbación aleatoria fuerte para garantizar el movimiento.
    # Genera un vector aleatorio en el rango [-0.5, 0.5]
    p_inicial += 1.0 * np.random.rand(2) - 0.5 
    print(f"-> Partícula 'empujada' a: ({p_inicial[0]:.4f}, {p_inicial[1]:.4f})")
# --------------------------------------------------------------------------

# Cálculo de la trayectoria usando el método de Euler (Descenso por Gradiente)
trayectoria_xy = [p_inicial]       # Lista para almacenar el historial de posiciones.
p_actual = p_inicial.copy()        # Posición actual de la partícula (se inicializa con el punto 'empujado').

for _ in range(10000):             # Bucle de iteración (máximo 10000 pasos).
    # Condición de parada: El bucle termina si la partícula sale del límite [-5, 5].
    if not (np.abs(p_actual[0]) < limite and np.abs(p_actual[1]) < limite):
        break                     # Detiene el bucle.
    
    grad = grad_f(p_actual[0], p_actual[1]) # Calcula el gradiente en la posición actual.
    
    # Método de Euler: Nuevo punto = Punto actual - (paso * gradiente)
    p_actual = p_actual - h * grad # El signo negativo asegura el movimiento en la dirección de máximo descenso.
    
    trayectoria_xy.append(p_actual) # Almacena la nueva posición.

trayectoria_xy = np.array(trayectoria_xy) # Convierte la lista final a un arreglo NumPy.


# ----------------------------------------------------------------------
# -- SECCIÓN 2: VISUALIZACIÓN (GRÁFICA 3D Y ANIMACIÓN) --
# ----------------------------------------------------------------------

# Preparar datos para la gráfica 3D (Superficie)
x_vals = np.linspace(-limite, limite, 100) # Crea 100 puntos espaciados uniformemente en X.
y_vals = np.linspace(-limite, limite, 100) # Crea 100 puntos espaciados uniformemente en Y.
X, Y = np.meshgrid(x_vals, y_vals)         # Genera la malla 2D (grid) para la superficie.
Z = f(X, Y)                                # Calcula la altura Z de la superficie en cada punto de la malla.

# Preparar datos para la trayectoria en 3D
trayectoria_x = trayectoria_xy[:, 0]        # Extrae todas las coordenadas X de la trayectoria.
trayectoria_y = trayectoria_xy[:, 1]        # Extrae todas las coordenadas Y de la trayectoria.
trayectoria_z = f(trayectoria_x, trayectoria_y) # Calcula la altura Z de la superficie en cada punto de la trayectoria.

# Configurar la figura y los ejes 3D para la animación
fig = plt.figure(figsize=(12, 9))          # Crea la ventana de la figura con un tamaño específico.
ax = fig.add_subplot(111, projection='3d') # Añade un solo set de ejes 3D a la figura.

# Dibuja la superficie del campo escalar.
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax.set_title("Animación de Partícula en Descenso por Gradiente", fontsize=16)

# Etiquetas de los ejes
ax.set_xlabel("Eje X"); ax.set_ylabel("Eje Y"); ax.set_zlabel("Eje Z")

# [OPCIONAL] Descomentar y ajustar para fijar el ángulo de visión de la cámara.
# ax.view_init(elev=30, azim=-60) 

# Crear los objetos gráficos que se van a animar
particula, = ax.plot([], [], [], 'ro', markersize=10, label='Partícula') # Objeto del punto rojo que se moverá.
rastro, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria') # Objeto de la línea roja que dibuja el rastro.
ax.legend() # Muestra la leyenda de los objetos.

def update(frame, tx, ty, tz):
    """Función que se llama repetidamente por FuncAnimation para actualizar el gráfico."""
    # Actualiza la posición de la partícula al punto del 'frame' actual.
    particula.set_data_3d([tx[frame]], [ty[frame]], [tz[frame]])
    # Actualiza el rastro, dibujando todos los puntos desde el inicio hasta el 'frame' actual.
    rastro.set_data_3d(tx[:frame+1], ty[:frame+1], tz[:frame+1])
    return particula, rastro # Retorna los objetos modificados.

# Crear y ejecutar la animación (FuncAnimation)
ani = animation.FuncAnimation(fig, update, frames=len(trayectoria_x), 
                              fargs=(trayectoria_x, trayectoria_y, trayectoria_z),
                              interval=2, blit=False, repeat=False)
# 'frames': Define el número total de cuadros (la longitud de la trayectoria).
# 'interval': Pausa entre cuadros en milisegundos (2 ms = animación muy rápida).


# ----------------------------------------------------------------------
# -- SECCIÓN 3: CÁLCULO NUMÉRICO DE LA INTEGRAL DE LÍNEA --
# ----------------------------------------------------------------------

integral_de_linea = 0.0 # Inicializa el acumulador.
for k in range(len(trayectoria_xy) - 1): # Itera sobre cada segmento de la trayectoria.
    p_k = trayectoria_xy[k]            # Punto de inicio del segmento.
    
    # Calcula la longitud del segmento (ds)
    p_k_plus_1 = trayectoria_xy[k+1]
    ds = np.linalg.norm(p_k_plus_1 - p_k) # Norma euclidiana (distancia) entre los dos puntos.
    
    # Evalúa el valor del campo escalar en el punto.
    f_val = f(p_k[0], p_k[1])
    
    # Aproximación Numérica: Suma (Valor del campo escalar * Longitud del segmento)
    integral_de_linea += f_val * ds

# Imprimir el resultado final en la consola al ejecutar el script
print("\n--- RESULTADOS NUMÉRICOS ---")
print(f"Punto inicial: ({p_inicial[0]:.4f}, {p_inicial[1]:.4f})")
print(f"Puntos generados en la trayectoria: {len(trayectoria_xy)}")
print(f"Punto final: ({trayectoria_xy[-1][0]:.4f}, {trayectoria_xy[-1][1]:.4f})")
print(f"Valor numérico de la integral de línea: {integral_de_linea:.4f}")


# ----------------------------------------------------------------------
# -- EJECUCIÓN FINAL --
# ----------------------------------------------------------------------

# Muestra la ventana con la gráfica 3D y ejecuta la animación.
plt.show()

# Opcional: Guardar la animación (descomentar para usar)
# print("Guardando animación...")
# ani.save('particula_descenso.mp4', writer='ffmpeg', fps=30)
# print("¡Animación guardada!")