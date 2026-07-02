"""
AegisQuant – Периодический мониторинг дрейфа и автоматическое переобучение.
Запускается по расписанию (cron / Task Scheduler) раз в день.
"""

import logging
import joblib
import pandas as pd
from DriftMonitor import DriftMonitor
from churn_model import run_training_pipeline
from config import ARTIFACT_PATH, CHURN_FEATURES, CLUSTER_FEATURES

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aegis_retrain")

# ── 1. Загрузка референсных данных ──────────────────────────────────────────
#     Обычно референсный датасет — это тренировочная выборка, на которой
#     обучалась модель. Её можно сохранить при обучении.
#     Здесь мы для примера загружаем сохранённый файл.
REFERENCE_DATA_PATH = "reference_data.pkl"

def load_production_features() -> pd.DataFrame:
    df = pd.read_csv("latest_production_data.csv")
    return df[CLUSTER_FEATURES + CHURN_FEATURES]

def load_or_create_reference() -> pd.DataFrame:
    """
    Загружает референсный датасет из файла.
    Если файла нет, генерирует синтетические данные и сохраняет их.
    """
    try:
        ref = pd.read_pickle(REFERENCE_DATA_PATH)
        log.info("Референсные данные загружены из %s", REFERENCE_DATA_PATH)
    except FileNotFoundError:
        log.warning("Файл референсных данных не найден, генерирую синтетику.")
        from data_pipeline import generate_synthetic_data
        df = generate_synthetic_data(n=3000)
        ref = df[CLUSTER_FEATURES + CHURN_FEATURES]
        ref.to_pickle(REFERENCE_DATA_PATH)
        log.info("Референсные данные сохранены в %s", REFERENCE_DATA_PATH)
    return ref

# ── 2. Загрузка свежих производственных данных ──────────────────────────────
def load_production_features() -> pd.DataFrame:
    """
    Загружает текущие признаки из продакшена.
    !!! Реализуйте здесь свою логику получения данных (БД, API, лог-файл).
    """
    # Заглушка: генерируем синтетические данные (замените на реальный источник)
    log.warning("Используется заглушка load_production_features() – замените на реальную загрузку!")
    from data_pipeline import generate_synthetic_data
    df = generate_synthetic_data(n=500)
    return df[CLUSTER_FEATURES + CHURN_FEATURES]

# ── 3. Основной цикл проверки ────────────────────────────────────────────────
def check_and_retrain():
    log.info("=" * 60)
    log.info("Запуск проверки дрейфа данных...")

    # Загружаем референс
    reference_data = load_or_create_reference()

    # Создаём монитор
    monitor = DriftMonitor(reference_data=reference_data)

    # Загружаем свежие данные
    current_batch = load_production_features()

    # Вычисляем дрейф
    results = monitor.compute_drift(current_batch)

    # Проверяем, нужно ли переобучать
    if monitor.should_retrain(results):
        log.warning("Дрейф обнаружен! Запуск переобучения модели AegisQuant...")
        run_training_pipeline()
        # После переобучения обновляем референсный датасет новой тренировочной выборкой
        from data_pipeline import generate_synthetic_data
        df = generate_synthetic_data(n=3000)
        new_ref = df[CLUSTER_FEATURES + CHURN_FEATURES]
        new_ref.to_pickle(REFERENCE_DATA_PATH)
        log.info("Референсный датасет обновлён.")
    else:
        log.info("Дрейф не обнаружен. Переобучение не требуется.")

if __name__ == "__main__":
    check_and_retrain()