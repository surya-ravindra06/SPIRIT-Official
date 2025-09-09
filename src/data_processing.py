import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import datetime
import io
import zipfile
import shutil
import glob
import pandas as pd
import pytz
import numpy as np
from pvlib import location
from scipy import optimize
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, HfFolder
from typing import Dict, Any
import yaml


class SolarDataProcessor:
    """Complete solar irradiance data processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.hf_config = config['huggingface']
        self.physics_config = config['physics']
        
    def extract_zip_url(self, url: str) -> str:
        """Extract ZIP file URL from NREL page."""
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            link = soup.find('a', href=re.compile(r'.*\.zip'))
            if link:
                return link.get('href')
        return None

    def download_and_extract_zip(self, zip_url: str, save_dir: str):
        """Download and extract ZIP file."""
        os.makedirs(save_dir, exist_ok=True)
        response = requests.get(zip_url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(save_dir)
        else:
            print(f"Failed to download ZIP file. Status code: {response.status_code}")

    def download_images_in_range(self, start_date: datetime.date, end_date: datetime.date, base_save_path: str):
        """Download images for specified date range."""
        current_date = start_date
        
        while current_date <= end_date:
            year, month, day = current_date.year, current_date.month, current_date.day
            url = f"https://midcdmz.nrel.gov/apps/imageranim.pl?site=SRRL;year={year};month={month};day={day};type="
            save_dir = os.path.join(base_save_path, f"{year}-{month:02d}-{day:02d}")
            
            print(f"Processing data for {current_date}")
            zip_url = self.extract_zip_url(url)
            if zip_url:
                zip_url = urljoin(url, zip_url)
                self.download_and_extract_zip(zip_url, save_dir)
            
            current_date += datetime.timedelta(days=1)

    def organize_images(self, src_directory: str, dest_directory: str):
        """Move all images to a single directory."""
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        folders = [os.path.join(src_directory, folder) for folder in os.listdir(src_directory) 
                  if os.path.isdir(os.path.join(src_directory, folder))]
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

        for folder in folders:
            for ext in image_extensions:
                images = glob.glob(os.path.join(folder, ext))
                for image in images:
                    try:
                        shutil.move(image, os.path.join(dest_directory, os.path.basename(image)))
                        print(f"Moved: {image}")
                    except Exception as e:
                        print(f"Error moving {image}: {e}")

    def process_and_rename_images(self, source_directory: str, raw_images_dir: str, processed_images_dir: str):
        """Process and rename images to standardized format."""
        os.makedirs(raw_images_dir, exist_ok=True)
        os.makedirs(processed_images_dir, exist_ok=True)

        raw_pattern = re.compile(r'^(\d{8})(\d{6})\.raw\.jpg$')
        pro_pattern = re.compile(r'^(\d{8})(\d{6})\.pro\.png$')

        for filename in os.listdir(source_directory):
            source_path = os.path.join(source_directory, filename)
            
            if raw_pattern.match(filename):
                match = raw_pattern.match(filename)
                datetime_str = match.group(1) + match.group(2)
                date_time = datetime.datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
                formatted_filename = date_time.strftime("%Y_%m_%d_%H_%M.jpg")
                destination_path = os.path.join(raw_images_dir, formatted_filename)
                shutil.copy(source_path, destination_path)
                
            elif pro_pattern.match(filename):
                match = pro_pattern.match(filename)
                datetime_str = match.group(1) + match.group(2)
                date_time = datetime.datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
                formatted_filename = date_time.strftime("%Y_%m_%d_%H_%M.png")
                destination_path = os.path.join(processed_images_dir, formatted_filename)
                shutil.copy(source_path, destination_path)

    def process_csv_data(self, csv_path: str, raw_images_dir: str, processed_images_dir: str) -> pd.DataFrame:
        """Process CSV data and link with images."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Create datetime column
        df['date_and_time'] = pd.to_datetime(df['DATE'] + ' ' + df['MST'], format='%m/%d/%Y %H:%M')
        df['Date_and_time'] = df['date_and_time'].dt.strftime('%Y_%m_%d_%H_%M')
        
        # Link raw images
        def find_raw_image(row):
            image_name = f"{row['Date_and_time']}.jpg"
            image_path = os.path.join(raw_images_dir, image_name)
            return image_path if os.path.exists(image_path) else None
        
        # Link processed images
        def find_processed_image(row):
            image_name = f"{row['Date_and_time']}.png"
            image_path = os.path.join(processed_images_dir, image_name)
            return image_path if os.path.exists(image_path) else None
        
        df['Raw_images'] = df.apply(find_raw_image, axis=1)
        df['Processed_Images'] = df.apply(find_processed_image, axis=1)
        
        # Remove rows with missing images
        df = df.dropna()
        
        return df

    def convert_mst_to_utc(self, mst_time_str: str) -> str:
        """Convert MST to UTC time."""
        mst = pytz.timezone('America/Denver')
        local_time_mst = mst.localize(pd.Timestamp(mst_time_str))
        utc_time = local_time_mst.astimezone(pytz.utc)
        return utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    def calculate_clear_sky_irradiance(self, date_time: str, latitude: float = None, longitude: float = None):
        """Calculate clear-sky irradiance values."""
        if latitude is None:
            latitude = self.physics_config['latitude']
        if longitude is None:
            longitude = self.physics_config['longitude']
            
        loc = location.Location(latitude, longitude)
        date_time_index = pd.DatetimeIndex([date_time])
        return loc.get_clearsky(date_time_index, model='ineichen')

    def physics_calculations(self, row: pd.Series) -> Dict[str, float]:
        """Perform physics-based calculations for solar panel optimization."""
        solar_elevation = 90 - row['Zenith_angle']
        solar_azimuth = row['Azimuth_angle']
        
        # Optimal panel orientation
        panel_tilt, panel_orientation = self.get_optimal_tilt_orientation(solar_elevation, solar_azimuth)
        
        # Angle of incidence
        aoi = np.rad2deg(np.arccos(-1 * self.max_func(
            np.deg2rad(panel_tilt), np.deg2rad(solar_elevation), 
            np.deg2rad(solar_azimuth), np.deg2rad(panel_orientation)
        )))
        
        # Irradiance calculations
        diffused_irradiance = self.diffused_irradiance_sandia(
            row['Clear_sky_dhi'], row['Clear_sky_ghi'], panel_tilt, row['Zenith_angle']
        )
        
        reflected_irradiance = self.reflected_irradiance(
            row['Clear_sky_ghi'], self.physics_config['albedo'], panel_tilt
        )
        
        direct_irradiance_tilted = self.tilted_irradiance_direct(row['Clear_sky_dni'], aoi)
        
        total_irradiance = self.total_irradiance_capping(
            self.physics_config['dc_ac_ratio'], direct_irradiance_tilted, 
            reflected_irradiance, diffused_irradiance
        )
        
        return {
            'physics_panel_tilt': panel_tilt,
            'physics_panel_orientation': panel_orientation,
            'physics_aoi': aoi,
            'physics_diffused_irradiance': diffused_irradiance,
            'physics_reflected_irradiance_tilted': reflected_irradiance,
            'physics_direct_irradiance_tilted': direct_irradiance_tilted,
            'physics_total_irradiance_tilted': total_irradiance
        }

    def max_func(self, panel_tilt: float, solar_elevation: float, solar_azimuth: float, panel_azimuth: float) -> float:
        """Objective function for panel optimization."""
        return -1 * (
            np.cos(solar_elevation) * np.sin(panel_tilt) * np.cos(panel_azimuth - solar_azimuth) +
            np.sin(solar_elevation) * np.cos(panel_tilt)
        )

    def get_optimal_tilt_orientation(self, solar_elevation: float, solar_azimuth: float):
        """Get optimal panel tilt and orientation."""
        panel_azimuth_east, panel_azimuth_west = 90, 270
        
        min_east = optimize.fmin(
            self.max_func, 1, 
            args=(np.deg2rad(solar_elevation), np.deg2rad(solar_azimuth), np.deg2rad(panel_azimuth_east)), 
            disp=False
        )[0]
        
        min_west = optimize.fmin(
            self.max_func, 1, 
            args=(np.deg2rad(solar_elevation), np.deg2rad(solar_azimuth), np.deg2rad(panel_azimuth_west)), 
            disp=False
        )[0]
        
        return (np.rad2deg(min_east), panel_azimuth_east) if min_east > min_west else (np.rad2deg(min_west), panel_azimuth_west)

    def diffused_irradiance_sandia(self, dhi: float, ghi: float, surface_tilt_angle: float, solar_zenith_angle: float) -> float:
        """Calculate diffused irradiance using Sandia model."""
        return (0.5 * dhi * (1 + np.cos(np.deg2rad(surface_tilt_angle))) + 
                0.5 * ghi * (0.012 * solar_zenith_angle - 0.04) * (1 - np.cos(np.deg2rad(surface_tilt_angle))))

    def reflected_irradiance(self, ghi: float, albedo: float, tilt: float) -> float:
        """Calculate reflected irradiance."""
        return 0.5 * (1 - np.cos(np.deg2rad(tilt))) * albedo * ghi

    def tilted_irradiance_direct(self, dni: float, aoi: float) -> float:
        """Calculate direct irradiance on tilted surface."""
        return dni * np.cos(np.deg2rad(aoi))

    def total_irradiance_capping(self, dc_ac_ratio: float, tilted_dni: float, tilted_ri: float, diffused_irradiance: float, threshold: float = 1000) -> float:
        """Calculate total irradiance with capping."""
        return np.minimum(dc_ac_ratio * (tilted_dni + tilted_ri + diffused_irradiance), threshold)

    def add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add clear-sky values and physics-based calculations."""
        enhanced_rows = []
        
        for _, row in df.iterrows():
            # Convert to UTC and calculate clear-sky
            date_str = row['DATE']
            time_str = row['MST']
            date_time_str = f"{date_str} {time_str}"
            formatted_date_time = datetime.datetime.strptime(date_time_str, "%m/%d/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
            utc_date_time = self.convert_mst_to_utc(formatted_date_time)
            
            # Calculate clear-sky irradiance
            irradiance_values = self.calculate_clear_sky_irradiance(utc_date_time)
            row['Clear_sky_ghi'] = irradiance_values['ghi'].iloc[0]
            row['Clear_sky_dni'] = irradiance_values['dni'].iloc[0]
            row['Clear_sky_dhi'] = irradiance_values['dhi'].iloc[0]
            
            # Add physics calculations
            physics_values = self.physics_calculations(row)
            for key, value in physics_values.items():
                row[key] = value
            
            enhanced_rows.append(row)
        
        return pd.DataFrame(enhanced_rows)

    def upload_to_huggingface(self, df: pd.DataFrame):
        """Upload processed dataset to Hugging Face."""
        # Prepare data for HF dataset
        data = {
            "DATE": df['DATE'].tolist(),
            "MST": df['MST'].tolist(),
            "Global_horizontal_irradiance": df['Global_horizontal_irradiance'].tolist(),
            "Direct_normal_irradiance": df['Direct_normal_irradiance'].tolist(),
            "Diffuse_horizontal_irradiance": df['Diffuse_horizontal_irradiance'].tolist(),
            "Air_temperature": df['Air_temperature'].tolist(),
            "Rel_humidity": df['Rel_humidity'].tolist(),
            "Avg_wind_speed": df['Avg_wind_speed'].tolist(),
            "Avg_wind_direction": df['Avg_wind_direction'].tolist(),
            "Pressure": df['Pressure'].tolist(),
            "Precipitation": df['Precipitation'].tolist(),
            "Zenith_angle": df['Zenith_angle'].tolist(),
            "Azimuth_angle": df['Azimuth_angle'].tolist(),
            "Clear_sky_ghi": df['Clear_sky_ghi'].tolist(),
            "Clear_sky_dni": df['Clear_sky_dni'].tolist(),
            "Clear_sky_dhi": df['Clear_sky_dhi'].tolist(),
            "physics_panel_tilt": df['physics_panel_tilt'].tolist(),
            "physics_panel_orientation": df['physics_panel_orientation'].tolist(),
            "physics_aoi": df['physics_aoi'].tolist(),
            "physics_diffused_irradiance": df['physics_diffused_irradiance'].tolist(),
            "physics_reflected_irradiance_tilted": df['physics_reflected_irradiance_tilted'].tolist(),
            "physics_direct_irradiance_tilted": df['physics_direct_irradiance_tilted'].tolist(),
            "physics_total_irradiance_tilted": df['physics_total_irradiance_tilted'].tolist(),
            "Raw_images": df['Raw_images'].tolist(),
            "Processed_images": df['Processed_Images'].tolist(),
        }
        
        # Define features
        features = Features({
            "DATE": Value("string"),
            "MST": Value("string"),
            "Global_horizontal_irradiance": Value("float32"),
            "Direct_normal_irradiance": Value("float32"),
            "Diffuse_horizontal_irradiance": Value("float32"),
            "Air_temperature": Value("float32"),
            "Rel_humidity": Value("float32"),
            "Avg_wind_speed": Value("float32"),
            "Avg_wind_direction": Value("float32"),
            "Pressure": Value("float32"),
            "Precipitation": Value("float32"),
            "Zenith_angle": Value("float32"),
            "Azimuth_angle": Value("float32"),
            "Clear_sky_ghi": Value("float32"),
            "Clear_sky_dni": Value("float32"),
            "Clear_sky_dhi": Value("float32"),
            "physics_panel_tilt": Value("float32"),
            "physics_panel_orientation": Value("float32"),
            "physics_aoi": Value("float32"),
            "physics_diffused_irradiance": Value("float32"),
            "physics_reflected_irradiance_tilted": Value("float32"),
            "physics_direct_irradiance_tilted": Value("float32"),
            "physics_total_irradiance_tilted": Value("float32"),
            "Raw_images": Image(),
            "Processed_images": Image(),
        })
        
        # Create and upload dataset
        dataset = Dataset.from_dict(data, features=features)
        
        HfFolder.save_token(self.hf_config['token'])
        dataset.push_to_hub(self.hf_config['dataset_name'])
        
        print(f"Dataset uploaded to Hugging Face: {self.hf_config['dataset_name']}")

    def process_complete_pipeline(self):
        """Execute the complete data processing pipeline."""
        print("Starting complete data processing pipeline...")
        
        # Parse dates
        start_date = datetime.datetime.strptime(self.data_config['start_date'], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(self.data_config['end_date'], "%Y-%m-%d").date()
        
        base_path = self.data_config['base_save_path']
        raw_data_path = os.path.join(base_path, "raw_data")
        images_path = os.path.join(base_path, "images")
        raw_images_path = os.path.join(base_path, self.data_config['raw_images_dir'])
        processed_images_path = os.path.join(base_path, self.data_config['processed_images_dir'])
        
        # Step 1: Download images
        print("Step 1: Downloading images...")
        self.download_images_in_range(start_date, end_date, raw_data_path)
        
        # Step 2: Organize images
        print("Step 2: Organizing images...")
        self.organize_images(raw_data_path, images_path)
        
        # Step 3: Process and rename images
        print("Step 3: Processing and renaming images...")
        self.process_and_rename_images(images_path, raw_images_path, processed_images_path)
        
        # Step 4: Process CSV data
        print("Step 4: Processing CSV data...")
        df = self.process_csv_data(self.data_config['main_csv_path'], raw_images_path, processed_images_path)
        
        # Step 5: Add enhanced features
        print("Step 5: Adding enhanced features...")
        df_enhanced = self.add_enhanced_features(df)
        
        # Step 6: Save processed dataset
        print("Step 6: Saving processed dataset...")
        output_path = os.path.join(base_path, self.data_config['final_dataset_name'])
        df_enhanced.to_csv(output_path, index=False)
        
        # Step 7: Upload to Hugging Face
        print("Step 7: Uploading to Hugging Face...")
        self.upload_to_huggingface(df_enhanced)
        
        print("Data processing pipeline completed successfully!")
        return df_enhanced