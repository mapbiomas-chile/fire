# last_update: '2025/01/26', github:'mapbiomas/chile-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_0_1_basic_authentication_and_parameterization.py
### Step A_0_1 - Basic authentication and parameterization

# =========================================
# 1. COUNTRY / COLLECTION SETTINGS
# =========================================
country = 'chile'

# Nombre lógico para modelos y logs
collection_name = 'col1'

# Nombre real de carpeta en GCS / estructura de datos
collection_folder = 'collection1'

# Subcarpeta base de la colección
base_subfolder = 'b24'

# Carpeta de mosaicos en GCS
mosaics_folder = 'mosaics_cog'

# Carpeta de modelos en GCS
models_folder = f'models_{collection_name}'

# Carpeta de salida en Earth Engine
ee_collection_folder = 'COLLECTION1'

# Proyecto y bucket
ee_project = f'mapbiomas-{country}'
bucket_name = 'mapbiomas-fire'

# Ruta base de datos en GCS
BASE_DATASET_PATH = f'{bucket_name}/sudamerica/{country}/{collection_folder}/{base_subfolder}'


# =========================================
# 2. AUTHENTICATION
# =========================================
def authenticates(ee_project, bucket_name):
    import ee
    from google.colab import auth
    from google.cloud import storage
    import gcsfs

    # Earth Engine
    ee.Authenticate()
    ee.Initialize(project=ee_project)

    # Google Cloud
    auth.authenticate_user()
    client = storage.Client(project=ee_project)
    bucket = client.bucket(bucket_name)

    # GCS filesystem
    fs = gcsfs.GCSFileSystem(project=ee_project)

    return client, bucket, fs


client, bucket, fs = authenticates(ee_project, bucket_name)


# =========================================
# 3. REPOSITORY PATHS
# =========================================
# Ajusta esta ruta según dónde clonaste realmente el repo
repo_root = '/content/mapbiomas-chile'

algorithms = f'{repo_root}/collection_010/classification_algorithms'

# Validación simple de ruta
import os
if not os.path.exists(algorithms):
    raise FileNotFoundError(f'No existe la carpeta algorithms: {algorithms}')


# =========================================
# 4. OPTIONAL PDF VIEWER
# =========================================
pdf_utils = '/content/brazil-fire/utils/google_collab_pdf_show.py'
pdf_path = '/content/brazil-fire/network/entrenamiento_de_monitoreo_de_cicatrices_de_fuego_en_regiones_de_la_red_mapBiomas.pdf'

if os.path.exists(pdf_utils) and os.path.exists(pdf_path):
    exec(open(pdf_utils).read())
    display_pdf_viewer(
        pdf_path,
        external_link="https://docs.google.com/presentation/d/1MPoqHWHLw-jJqKUStikJ0Cc-8oLuMuKZ4_c_kQA3DKQ"
    )
else:
    print('PDF viewer no cargado: no se encontró el repo/path de brazil-fire.')
