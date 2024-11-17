import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import logging
import glob
from tqdm import tqdm

class SensorPreprocessor:
    def __init__(self):
        self.setup_logging()
        self.scaler = StandardScaler()
    
    def setup_logging(self):

        # Loglama ayarlarını yapılandırır
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sensor_preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_sensor_data(self, file_path, sensor_type):
        
        # Sensör verilerini yükler ve formatlar
        
        try:
            if sensor_type == "Accelerometer" or sensor_type == "Gyroscope":
                columns = ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation']
            else:
                raise ValueError(f"Bilinmeyen sensör tipi: {sensor_type}")

            df = pd.read_csv(file_path, header=None, names=columns)
            numeric_columns = [col for col in df.columns if col not in ['ActivityID', 'Phone_orientation']]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            return df
        
        except Exception as e:
            self.logger.error(f"Dosya yüklenirken hata: {file_path} - {str(e)}")
            return None

    def remove_outliers(self, df, columns, n_std=3):
        
        # Aykırı değerleri temizler
        
        df_clean = df.copy()
        for column in columns:
            z_scores = np.abs(stats.zscore(df_clean[column].dropna()))
            df_clean = df_clean[z_scores < n_std]

            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[column] >= Q1 - 1.5 * IQR) & 
                (df_clean[column] <= Q3 + 1.5 * IQR)
            ]
        return df_clean

    def normalize_features(self, df, columns):
        
        # Verileri normalize eder
        
        df_normalized = df.copy()
        if not df.empty:
            df_normalized[columns] = self.scaler.fit_transform(df[columns])
        else:
            self.logger.warning("Normalizasyon sırasında boş veri setiyle karşılaşıldı.")
        return df_normalized

    def process_sensor_data(self, input_file, output_file, sensor_type):
        
        # Tek bir sensör dosyası için tüm preprocessing adımlarını uygular
        
        try:
            df = self.load_sensor_data(input_file, sensor_type)
            if df is None or df.empty:
                self.logger.warning(f"Dosya boş veya yüklenemedi: {input_file}")
                return None

            if sensor_type == "Accelerometer" or sensor_type == "Gyroscope":
                sensor_columns = ['X', 'Y', 'Z']
            else:
                self.logger.error(f"Bilinmeyen sensör tipi: {sensor_type}")
                return None

            df_clean = self.remove_outliers(df, sensor_columns)
            self.logger.info(f"Aykırı değer temizleme sonrası veri boyutu: {df_clean.shape}")
            if df_clean.empty:
                self.logger.warning(f"Aykırı değer temizleme sonrası veri seti boş: {input_file}")
                return None

            df_normalized = self.normalize_features(df_clean, sensor_columns)
            if df_normalized.empty:
                self.logger.warning(f"Normalizasyon sonrası veri seti boş: {input_file}")
                return None

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_normalized.to_csv(output_file, index=False)
            self.logger.info(f"İşlenmiş veri kaydedildi: {output_file}")
            return df_normalized

        except Exception as e:
            self.logger.error(f"Dosya işlenirken hata: {input_file} - {str(e)}")
            return None

def process_all_sensors():
    
    # Tüm users ve sessions için sensör verilerini işler
    
    preprocessor = SensorPreprocessor()
    base_path = "dataset"
    output_base_path = "processed_data"
    sensor_types = ["Accelerometer", "Gyroscope"]

    user_folders = glob.glob(os.path.join(base_path, "*"))
    
    for user_folder in tqdm(user_folders, desc="Users"):
        user_id = os.path.basename(user_folder)
        
        session_folders = glob.glob(os.path.join(user_folder, "*_session_*"))
        
        for session_folder in tqdm(session_folders, desc=f"Sessions for {user_id}"):
            session_id = os.path.basename(session_folder)
            
            for sensor_type in sensor_types:
                input_file = os.path.join(session_folder, f"{sensor_type}.csv")
                output_file = os.path.join(output_base_path, user_id, session_id, f"processed_{sensor_type.lower()}.csv")
                
                if os.path.exists(input_file):
                    preprocessor.logger.info(f"İşleniyor: {input_file}")
                    preprocessor.process_sensor_data(input_file, output_file, sensor_type)
                else:
                    preprocessor.logger.warning(f"Dosya bulunamadı: {input_file}")

def main():
    print("Sensör verilerinin toplu process'i başlıyor..")
    process_all_sensors()
    print("Process tamamlandı..")

if __name__ == "__main__":
    main()
