<p align="center">
  <img src="docs/assets/logo_mapbiomas_Fuego.png" alt="MapBiomas Fuego" width="360"/>
</p>

# MapBiomas Fire — Chile (`chile-fire`)

**Español** | [English](README.md)

Herramientas de mapeo de **área quemada en Chile** en el marco de la colección MapBiomas Fire. El repositorio implementa un flujo de trabajo completo basado en archivos, ejecutable en máquina local o en HPC (NLHPC leftraru): entrenamiento con muestras etiquetadas, inferencia sobre mosaicos, postfiltro por cobertura de suelo (LULC), vectorización y validación espacial frente a catálogos de cicatrices de referencia.

**Campaña de referencia de producción:** rutas y parámetros etiquetados `20260619` (receta conservadora de red tipo *multilayer perceptron*). Actualice las rutas al promover un nuevo directorio de campaña.

---

## 1. Flujo científico (visión general)

```text
Muestras de entrenamiento (GeoTIFF etiquetados)
        │
        ▼
[1] classification/   Entrena MLP por región × periodo → checkpoint
        │
        ▼
Mosaicos estacionales / anuales (GeoTIFF de predictores)
        │
        ▼
[1] classification/   Inferencia → rasters clasificados binarios
        │
        ▼
[2] filtering/        Consistencia temporal, relleno de huecos, máscaras LULC
        │
        ▼
[3] vectorize/        Raster → polígonos (tesela y/o escala nacional)
        │
        ▼
[2|3] filtro de área  Tamaño mínimo de evento / reglas percentílicas (dominio vectorial)
        │
        ▼
[4] validation/       Métricas en proyección de área igual vs cicatrices de referencia
```

| Etapa | Módulo | Función |
|-------|--------|---------|
| 1 | [`classification/`](classification/README.es.md) | Clasificación supervisada quemado / no quemado (MLP TensorFlow) |
| 2 | [`filtering/`](filtering/README.es.md) | Limpieza post-clasificación con LULC MapBiomas |
| 3 | [`vectorize/`](vectorize/README.es.md) | Poligonización y agrupación nacional de eventos |
| 4 | [`validation/`](validation/README.es.md) | Alineación de referencia, intersección, Jaccard / métricas de Singh |
| — | [`lib/`](lib/README.es.md) | Utilidades Python compartidas |
| — | [`utilities/`](utilities/README.es.md) | Exportación GEE, listado de tiles, mosaicos |
| — | [`experiments/`](experiments/README.es.md) | Configuraciones A/B archivadas (no producción) |
| — | `collection_010/` | Materiales legados de la Collection 0.1.0 |

---

## 2. Requisitos para la replicación

| Requisito | Notas |
|-----------|--------|
| Python 3.11+ | Conda con `numpy`, `rasterio`, `scipy`, `tensorflow` (modo compat 1.x), `geopandas`, `shapely`, `pandas`, `matplotlib` |
| Muestras GeoTIFF | Pilas con banda de etiqueta `landcover` |
| Mosaicos anuales | Mismas bandas de predictores que el entrenamiento |
| Stack LULC MapBiomas | GeoTIFF multibanda (no VRT) para generación de máscaras |
| HPC (opcional) | SLURM en NLHPC; muchos post-pasos se ejecutan en nodo de *login* |

La **configuración de rutas** no se fija en el código de producción. Copie `cluster_paths.*.env.example` o las plantillas `*.leftraru` a un `cluster_paths.env` **local** (ignorado por git), edite rutas absolutas y ejecute `source` antes de lanzar los procesos.

---

## 3. Secuencia recomendada de extremo a extremo (NLHPC)

Adapte usuario y carpetas de campaña a su cuenta.

```bash
cd ~/fire
git fetch origin && git checkout main && git pull

# Rutas unificadas de producción (entrenamiento + clasificación + filtro)
cp classification/cluster_paths.20260619.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

# (A) Entrenar siete *checkpoints* de Chile (omitir si el modelo ya existe)
bash classification/run_train_chile_campaign.sh

# (B) Clasificar la serie completa r1/r2/r4/r6 × 2013–2025
sbatch --export=ALL classification/run_classify_chile_slurm.sh

# (C) Filtro LULC + temporal + relleno de huecos
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh

# (D) Vectorizar teselas → filtro de área → productos nacionales opcionales
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
source vectorize/cluster_paths.env
bash vectorize/run_post_filter_pipeline.sh
```

La documentación de cada módulo detalla parámetros suficientes para revisión por pares y reimplementación independiente.

---

## 4. Convención de versiones del modelo (Chile)

Los *checkpoints* se nombran  
`col1_chile_<version>_<region>_rnn_lstm_ckpt`.

| Región | 2013–2018 | 2019–2025 |
|--------|-----------|-----------|
| r2 | v1 | v1 |
| r1, r4, r6 | v1 | v2 |

Las muestras de entrenamiento suelen usar el token `samples_fire_v1_*` (`SAMPLE_VERSION=v1`). La **versión del *checkpoint*** se elige por el **rango de años** del subconjunto de entrenamiento, no solo por ese token.

---

## 5. Lista de control de reproducibilidad

- [ ] Registrar el SHA del commit (`git rev-parse HEAD`) en cada campaña  
- [ ] Archivar `*_hyperparameters.json` junto a cada *checkpoint* TensorFlow  
- [ ] Conservar el `cluster_paths.env` exacto usado (no solo la plantilla)  
- [ ] Documentar umbral de decisión (producción **0,55**), pérdida, semilla y modo de partición train/val  
- [ ] Identificar el stack LULC, año de la banda 1, ventanas de estabilidad y *flags* de filtrado  
- [ ] Registrar tamaños de *sieve* y distancias de agrupación al publicar eventos nacionales  

**Guía operativa ampliada** (lista de artefactos, convención de nombres de productos y plantillas de citación commit + entorno):

- [docs/CAMPAIGN_TRACEABILITY.es.md](docs/CAMPAIGN_TRACEABILITY.es.md) (español)  
- [docs/CAMPAIGN_TRACEABILITY.md](docs/CAMPAIGN_TRACEABILITY.md) (inglés)

---

## 6. Idioma

| Documento | Idioma |
|-----------|--------|
| [`README.md`](README.md) | Inglés |
| [`README.es.md`](README.es.md) | Español (este archivo) |
| Módulos `README.md` / `README.es.md` | Inglés / español por paquete |
| [`docs/CAMPAIGN_TRACEABILITY.es.md`](docs/CAMPAIGN_TRACEABILITY.es.md) | Artefactos, nombres, citación (ES) |
| [`docs/CAMPAIGN_TRACEABILITY.md`](docs/CAMPAIGN_TRACEABILITY.md) | Mismo contenido (EN) |

Ambas versiones son equivalentes: parámetros y secuencias de comandos deben coincidir.
