import os
from datetime import datetime
import pytz
import subprocess
import json
import psutil
import shutil

# =========================================
# GLOBAL VARIABLES
# =========================================
log_file_path_local = None
bucket_log_folder = None
log_index = 0
header_dict = None

# =========================================
# SAFE DEFAULTS FOR REQUIRED GLOBALS
# =========================================
if 'country' not in globals():
    country = 'chile'

if 'collection_name' not in globals():
    collection_name = 'col1'

if 'source_name' not in globals():
    source_name = 'IPAM'

if 'specified_timezone' not in globals():
    specified_timezone = None

if 'bucket_name' not in globals():
    bucket_name = 'mapbiomas-fire'

# Opcional: si quieres usar una base local explícita
LOCAL_LOG_BASE = f'/content/{bucket_name}/sudamerica/{country}/classification_logs'
GCS_LOG_BASE = f'gs://{bucket_name}/sudamerica/{country}/classification_logs'

# =========================================
# TIMEZONE CONFIGURATION
# =========================================
timezone_switch = {
    'brazil': 'America/Sao_Paulo',
    'guyana': 'America/Guyana',
    'bolivia': 'America/La_Paz',
    'colombia': 'America/Bogota',
    'chile': 'America/Santiago',
    'peru': 'America/Lima',
    'paraguay': 'America/Asuncion'
}

try:
    if specified_timezone:
        country_tz = pytz.timezone(specified_timezone)
    else:
        country_tz = pytz.timezone(timezone_switch.get(country, 'UTC'))
except Exception:
    country_tz = pytz.timezone('UTC')

# =========================================
# HEADER
# =========================================
def create_header():
    global header_dict

    initial_date = datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')
    timezone_str = specified_timezone if specified_timezone else timezone_switch.get(country, 'UTC')

    header_str = (
        f"Processing started on: {initial_date}\n"
        f"Timezone: {timezone_str}\n"
        f"Country: {country}\n"
        f"Source: {source_name}\n"
        f"Collection: {collection_name}\n"
        "---------------------------------\n"
    )

    header_dict = {
        "initial_date": initial_date,
        "timezone": timezone_str,
        "country": country,
        "source": source_name,
        "collection": collection_name
    }

    print(header_str)
    return header_str

# =========================================
# SYSTEM INFO
# =========================================
def get_system_info_compact():
    ram_info = psutil.virtual_memory()
    total_ram = ram_info.total / (1024 ** 3)
    available_ram = ram_info.available / (1024 ** 3)

    disk_info = shutil.disk_usage('/')
    total_disk = disk_info.total / (1024 ** 3)
    free_disk = disk_info.free / (1024 ** 3)

    return f"disk:{free_disk:.1f}/{total_disk:.1f}GB, ram:{available_ram:.1f}/{total_ram:.1f}GB"

# =========================================
# PATHS
# =========================================
def create_log_paths(timestamp):
    log_folder = LOCAL_LOG_BASE
    log_file_name = f'burned_area_classification_log_{collection_name}_{country}_{timestamp}.log'
    log_file_path_local = os.path.join(log_folder, log_file_name)
    bucket_log_folder = f'{GCS_LOG_BASE}/{log_file_name}'

    return log_folder, log_file_path_local, bucket_log_folder

def create_local_directory(log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
        print(f"[LOG INFO] Created local log directory: {log_folder}")
    else:
        print(f"[LOG INFO] Local log directory already exists: {log_folder}")

# =========================================
# LOG FORMAT
# =========================================
def format_log_entry(message, log_index, system_info):
    if isinstance(message, (dict, list)):
        message = json.dumps(message, default=str)
    elif not isinstance(message, str):
        message = str(message)

    current_time = datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')

    log_entry = {
        "index": log_index,
        "timestamp": current_time,
        "message": message,
        "system_info": system_info,
        "header": header_dict
    }
    return json.dumps(log_entry, ensure_ascii=False) + "\n"

def write_log_local(log_file_path_local, log_entry):
    with open(log_file_path_local, 'a', encoding='utf-8') as log_file:
        log_file.write(log_entry)

def upload_log_to_gcs(log_file_path_local, bucket_log_folder):
    try:
        subprocess.check_call(f'gsutil cp "{log_file_path_local}" "{bucket_log_folder}"', shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[LOG ERROR] Failed to upload log file to GCS: {str(e)}")

# =========================================
# MAIN LOGGER
# =========================================
def log_message(message):
    global log_file_path_local, bucket_log_folder, log_index, header_dict

    if log_file_path_local is None:
        timestamp = datetime.now(country_tz).strftime('%Y-%m-%d_%H-%M-%S')
        log_folder, log_file_path_local, bucket_log_folder = create_log_paths(timestamp)

        create_local_directory(log_folder)

        header_str = create_header()
        with open(log_file_path_local, 'w', encoding='utf-8') as log_file:
            log_file.write(header_str)

    log_index += 1
    system_info = get_system_info_compact()
    log_entry = format_log_entry(message, log_index, system_info)

    formatted_log = f"[LOG] [{log_index}] [{datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')}] {message} | {system_info}"
    print(formatted_log)

    write_log_local(log_file_path_local, log_entry)
    upload_log_to_gcs(log_file_path_local, bucket_log_folder)
