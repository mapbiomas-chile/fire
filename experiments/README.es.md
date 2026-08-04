# experiments/

**Español** | [English](README.md)

Archivos de **configuración** archivados de corridas A/B de desarrollo. Se conservan para la **reproducibilidad metodológica** de decisiones de diseño y **no** son necesarios en la ruta de producción actual si existen los artefactos en `models_col1_20260619` / `classification_20260619`.

---

## Campaña conservadora (promovida a 20260619)

Estos entornos entrenaron y clasificaron con una receta conservadora (sin ventana espacial, sin *oversampling* de quemado, umbral fijo **0,55**, métrica IoU) para diagnosticar la sobreestimación de área quemada respecto a la campaña `classification_20260618`. La receta se promovió a producción como **`classification_20260619`**.

| Archivo | Rol histórico |
|---------|----------------|
| `classification/cluster_paths.train_conservative.env.leftraru` | Entrenar → `models_col1_conservative` |
| `classification/cluster_paths.classify_conservative.env.leftraru` | Clasificar → `classification_conservative` |
| `filtering/cluster_paths.conservative.env.leftraru` | Filtrar ese árbol |

**Env de producción (misma receta, rutas canónicas):**  
`classification/cluster_paths.20260619.env.leftraru` y `filtering/cluster_paths.20260619.env.leftraru`.

Véase [README de clasificación](../classification/README.es.md).
