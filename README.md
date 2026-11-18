# Mecanismo de Atención y Transformer básico para series de tiempo

Implementación práctica de un mecanismo de self-attention y de un Transformer básico (PyTorch) aplicados a una
serie de tiempo sintética, como parte de la actividad **R2-A3-S12 Transformers** (Deep Learning - Conceptos).

## Contenido

- `Transformers_Series_Tiempo.ipynb`: notebook completo (Google Colab / Jupyter), con preparación de datos,
  implementación del mecanismo de atención desde cero, implementación de un Transformer encoder, entrenamiento,
  evaluación y análisis de resultados.
- `experimento.py`: script equivalente en Python plano, usado para generar los resultados y gráficas.
- `resultados/`: gráficas generadas (serie original, curvas de pérdida, mapas de atención, predicciones vs.
  valores reales, comparación final) y `resumen.txt` con las métricas numéricas.

## Resultados principales

| Modelo | MSE en test |
|---|---|
| Atención simple (una cabeza) | 0.191 |
| Transformer (multi-cabeza + feed-forward + codificación posicional) | 0.021 |

## Cómo ejecutar

```bash
pip install torch numpy matplotlib
python experimento.py
```

o abrir `Transformers_Series_Tiempo.ipynb` en Google Colab.
