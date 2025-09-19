# 📌 Proyecto 1: Cálculo para Inteligencia Artificial  

Este proyecto implementa un **descenso por gradiente** sobre un campo escalar bidimensional y visualiza la trayectoria de una partícula en una superficie 3D.  
Además, calcula de manera numérica la **integral de línea** a lo largo de la trayectoria generada.  

---

## 📂 Contenido del proyecto
- **main.py** → Script principal con:
  - Definición de la función escalar `f(x,y)`.
  - Cálculo del gradiente ∇f(x,y).
  - Implementación del método de Euler para simular la trayectoria.
  - Visualización en 3D con animación usando `matplotlib`.
  - Cálculo de la integral de línea sobre la trayectoria.

---

## ⚙️ Requisitos
Este proyecto está hecho en **Python 3**.  
Instala las dependencias con:
```bash
pip install -r requirements.txt