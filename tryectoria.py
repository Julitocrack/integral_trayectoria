# PROYECTO 1: CÁLCULO PARA INTELIGENCIA ARTIFICIAL
# NOMBRE: Tryectoria
# AUTOR: Julio Rangel

# -- SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS --
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# -- SECCIÓN 1: MÉTODO MATEMÁTICO Y CÁLCULO DE TRAYECTORIA --

def f(x, y):
    """Define el campo escalar f(x,y)."""
    return x**2 * np.cos(y) - y**2 * np.sin(x)

def grad_f(x, y):
    """Calcula el gradiente de f(x,y) en un punto (x,y)."""
    df_dx = 2*x*np.cos(y) - y**2*np.cos(x)
    df_dy = -x**2*np.sin(y) - 2*y*np.sin(x)
    return np.array([df_dx, df_dy])

# Parámetros de la simulación
p_inicial = np.array([3.0, 3.0])
h = 0.001
limite = 5.0

# Cálculo de la trayectoria usando el método de Euler
trayectoria_xy = [p_inicial]
p_actual = p_inicial.copy()

for _ in range(5000):
    if not (np.abs(p_actual[0]) < limite and np.abs(p_actual[1]) < limite):
        break
    grad = grad_f(p_actual[0], p_actual[1])
    p_actual = p_actual - h * grad # Descenso por gradiente
    trayectoria_xy.append(p_actual)

trayectoria_xy = np.array(trayectoria_xy)


# -- SECCIÓN 2: VISUALIZACIÓN (GRÁFICA 3D Y ANIMACIÓN) --


# Preparar datos para la gráfica 3D
x_vals = np.linspace(-limite, limite, 100)
y_vals = np.linspace(-limite, limite, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

trayectoria_x = trayectoria_xy[:, 0]
trayectoria_y = trayectoria_xy[:, 1]
trayectoria_z = f(trayectoria_x, trayectoria_y)

# Configurar la figura y los ejes 3D para la animación
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax.set_title("Animación de Partícula en Descenso por Gradiente", fontsize=16)
ax.set_xlabel("Eje X"); ax.set_ylabel("Eje Y"); ax.set_zlabel("Eje Z")


# Crear los objetos gráficos que se van a animar
particula, = ax.plot([], [], [], 'ro', markersize=10, label='Partícula')
rastro, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria')
ax.legend()

def update(frame, tx, ty, tz):
    """Función que actualiza la animación en cada cuadro."""
    particula.set_data_3d([tx[frame]], [ty[frame]], [tz[frame]])
    rastro.set_data_3d(tx[:frame+1], ty[:frame+1], tz[:frame+1])
    return particula, rastro

# Crear y ejecutar la animación
ani = animation.FuncAnimation(fig, update, frames=len(trayectoria_x), 
                              fargs=(trayectoria_x, trayectoria_y, trayectoria_z),
                              interval=2, blit=False, repeat=False)



# -- SECCIÓN 3: CÁLCULO NUMÉRICO DE LA INTEGRAL DE LÍNEA --


integral_de_linea = 0.0
for k in range(len(trayectoria_xy) - 1):
    p_k = trayectoria_xy[k]
    p_k_plus_1 = trayectoria_xy[k+1]
    ds = np.linalg.norm(p_k_plus_1 - p_k)
    f_val = f(p_k[0], p_k[1])
    integral_de_linea += f_val * ds

# Imprimir el resultado final en la consola al ejecutar el script
print("\n--- RESULTADOS NUMÉRICOS ---")
print(f"Punto inicial: ({p_inicial[0]}, {p_inicial[1]})")
print(f"Puntos generados en la trayectoria: {len(trayectoria_xy)}")
print(f"Punto final: ({trayectoria_xy[-1][0]:.4f}, {trayectoria_xy[-1][1]:.4f})")
print(f"Valor numérico de la integral de línea: {integral_de_linea:.4f}")



# -- EJECUCIÓN FINAL --


# Muestra la ventana con la animación
plt.show()

# Opcional: Guardar la animación (requiere ffmpeg)
#print("Guardando animación...")
#ani.save('particula_descenso.mp4', writer='ffmpeg', fps=30)
#print("¡Animación guardada!")

