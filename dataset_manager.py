# Dataset Manager for Plant Disease Recognition
# Handles Kaggle dataset download and organization

import os
import zipfile
import subprocess
import sys
from pathlib import Path
import shutil
from typing import Tuple, Optional
import requests
from tqdm import tqdm


class KaggleDatasetManager:
    """
    Manages Kaggle dataset download and organization for plant disease recognition
    """

    def __init__(self, dataset_name: str = "rashikrahmanpritom/plant-disease-recognition-dataset"):
        self.dataset_name = dataset_name
        self.base_dir = Path("datasets")
        self.dataset_dir = self.base_dir / "plant-disease-recognition"
        self.zip_file = self.base_dir / "plant-disease-dataset.zip"

    def setup_kaggle_api(self) -> bool:
        """
        Setup Kaggle API credentials and install if needed
        """
        try:
            # Check if kaggle is installed
            import kaggle
            print("✅ Kaggle API already installed")
            return True
        except ImportError:
            print("📦 Installing Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
            print("✅ Kaggle API installed successfully")
            return True
        except Exception as e:
            print(f"❌ Error setting up Kaggle API: {e}")
            return False

    def check_kaggle_credentials(self) -> bool:
        """
        Check if Kaggle API credentials are properly configured
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle credentials verified")
            return True
        except Exception as e:
            print("❌ Kaggle credentials not found or invalid")
            print("\n🔧 To setup Kaggle credentials:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)")
            return False

    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download the plant disease dataset from Kaggle
        """
        if not force_download and self.dataset_dir.exists():
            print("✅ Dataset already exists")
            return True

        if not self.setup_kaggle_api():
            return False

        if not self.check_kaggle_credentials():
            return False

        try:
            # Create directories
            self.base_dir.mkdir(exist_ok=True)

            print(f"📥 Downloading dataset: {self.dataset_name}")

            # Import kaggle after installation
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            # Download dataset with correct format
            api.dataset_download_files(
                self.dataset_name,
                path=str(self.base_dir),
                unzip=True  # Changed to True for automatic extraction
            )

            print(f"✅ Dataset downloaded and extracted successfully")

            # Find the extracted directory
            extracted_dirs = [d for d in self.base_dir.iterdir() if
                              d.is_dir() and d.name != 'plant-disease-recognition']

            if extracted_dirs:
                # Rename to standard directory name
                source_dir = extracted_dirs[0]
                if source_dir != self.dataset_dir:
                    if self.dataset_dir.exists():
                        shutil.rmtree(self.dataset_dir)
                    source_dir.rename(self.dataset_dir)
                    print(f"✅ Renamed {source_dir.name} to {self.dataset_dir.name}")

            return True

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("🔍 Troubleshooting steps:")
            print(
                "1. Verify dataset URL: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset")
            print("2. Check if dataset is public and accessible")
            print("3. Verify Kaggle API credentials are correct")
            print("4. Try downloading manually from Kaggle website")
            return False

    def extract_dataset(self) -> bool:
        """
        Extract and organize the dataset
        """
        if not self.zip_file.exists():
            print("❌ Dataset zip file not found")
            return False

        try:
            print("📂 Extracting dataset...")

            # Extract zip file
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)

            # Find extracted folder
            extracted_folders = [f for f in self.base_dir.iterdir()
                                 if f.is_dir() and f.name != "plant-disease-recognition"]

            if extracted_folders:
                # Rename to standard name
                if extracted_folders[0] != self.dataset_dir:
                    if self.dataset_dir.exists():
                        shutil.rmtree(self.dataset_dir)
                    extracted_folders[0].rename(self.dataset_dir)

            print("✅ Dataset extracted successfully")
            return True

        except Exception as e:
            print(f"❌ Error extracting dataset: {e}")
            return False

    def organize_dataset(self) -> bool:
        """
        Organize dataset into proper structure using the larger Train directory
        """
        try:
            print("📁 Organizing dataset structure...")

            # Check what directories exist after extraction
            print("🔍 Scanning extracted directories...")

            # Look for the main Train directory with more images
            train_dir = self.base_dir / "Train" / "Train"
            validation_dir = self.base_dir / "Validation" / "Validation"

            if train_dir.exists():
                print(f"✅ Found main training directory with more images: {train_dir}")

                # Use the Train directory as our main dataset
                target_dir = self.dataset_dir
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)

                # Expected classes
                expected_classes = ['Healthy', 'Powdery', 'Rust']

                for class_name in expected_classes:
                    source_class_dir = train_dir / class_name
                    target_class_dir = target_dir / class_name

                    if source_class_dir.exists():
                        # Copy all images from train directory
                        target_class_dir.mkdir(exist_ok=True)
                        train_images = list(source_class_dir.glob("*"))

                        for img in train_images:
                            if img.is_file():
                                shutil.copy2(img, target_class_dir)

                        # Also add validation images if they exist
                        val_class_dir = validation_dir / class_name
                        if val_class_dir.exists():
                            val_images = list(val_class_dir.glob("*"))
                            for img in val_images:
                                if img.is_file():
                                    # Add prefix to avoid name conflicts
                                    new_name = f"val_{img.name}"
                                    shutil.copy2(img, target_class_dir / new_name)

                        total_images = len(list(target_class_dir.glob("*")))
                        print(f"✅ Organized {class_name}: {total_images} images")
                    else:
                        print(f"⚠️  Class not found in Train directory: {class_name}")

                print("✅ Dataset organization completed using full dataset")
                return True
            else:
                # Fallback to original smaller dataset
                print("⚠️  Large Train directory not found, using smaller dataset")
                return self._organize_small_dataset()

        except Exception as e:
            print(f"❌ Error organizing dataset: {e}")
            return False

    def _organize_small_dataset(self) -> bool:
        """
        Fallback method for smaller dataset organization
        """
        try:
            expected_classes = ['Healthy', 'Powdery', 'Rust']

            # Check if classes exist in the smaller dataset
            class_dirs = []
            for class_name in expected_classes:
                class_dir = self.dataset_dir / class_name
                if class_dir.exists():
                    image_count = len(list(class_dir.glob("*")))
                    class_dirs.append(class_dir)
                    print(f"✅ Found class: {class_name} ({image_count} images)")

            return len(class_dirs) >= 2

        except Exception as e:
            print(f"❌ Error organizing small dataset: {e}")
            return False

    def _cleanup_empty_dirs(self):
        """Remove empty directories"""
        for root, dirs, files in os.walk(self.dataset_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except:
                    pass

    def get_dataset_info(self) -> dict:
        """
        Get information about the organized dataset
        """
        info = {
            'dataset_path': str(self.dataset_dir),
            'classes': [],
            'total_images': 0,
            'class_distribution': {}
        }

        if not self.dataset_dir.exists():
            def debug_dataset_structure(self):

                """
            Debug method to understand the dataset structure
            """
        print("\n🔍 DEBUG: Dataset Structure Analysis")
        print("-" * 50)

        if not self.base_dir.exists():
            print("❌ Base directory doesn't exist")
            return

        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Expected dataset directory: {self.dataset_dir}")

        # List all items in base directory
        print("\n📋 Contents of base directory:")
        for item in self.base_dir.iterdir():
            item_type = "📁" if item.is_dir() else "📄"
            print(f"  {item_type} {item.name}")

        # If dataset dir exists, explore it
        if self.dataset_dir.exists():
            print(f"\n📋 Contents of {self.dataset_dir.name}:")
            for item in self.dataset_dir.iterdir():
                item_type = "📁" if item.is_dir() else "📄"
                if item.is_dir():
                    file_count = len(list(item.glob("*")))
                    print(f"  {item_type} {item.name} ({file_count} items)")
                else:
                    print(f"  {item_type} {item.name}")

        # Search for any directories that might contain images
        print(f"\n🔍 Searching for image directories...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for root, dirs, files in os.walk(self.base_dir):
            image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
            if image_files:
                rel_path = Path(root).relative_to(self.base_dir)
                print(f"  📸 {rel_path}: {len(image_files)} images")

        print("-" * 50)

        for class_dir in self.dataset_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(
                    class_dir.glob('*.jpeg'))
                num_images = len(image_files)

                info['classes'].append(class_name)
                info['class_distribution'][class_name] = num_images
                info['total_images'] += num_images

        return info

    def debug_dataset_structure(self):
        """
        Debug method to understand the dataset structure
        """
        print("\n🔍 DEBUG: Dataset Structure Analysis")
        print("-" * 50)

        if not self.base_dir.exists():
            print("❌ Base directory doesn't exist")
            return

        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Expected dataset directory: {self.dataset_dir}")

        # List all items in base directory
        print("\n📋 Contents of base directory:")
        for item in self.base_dir.iterdir():
            item_type = "📁" if item.is_dir() else "📄"
            print(f"  {item_type} {item.name}")

        # If dataset dir exists, explore it
        if self.dataset_dir.exists():
            print(f"\n📋 Contents of {self.dataset_dir.name}:")
            for item in self.dataset_dir.iterdir():
                item_type = "📁" if item.is_dir() else "📄"
                if item.is_dir():
                    file_count = len(list(item.glob("*")))
                    print(f"  {item_type} {item.name} ({file_count} items)")
                else:
                    print(f"  {item_type} {item.name}")

        # Search for any directories that might contain images
        print(f"\n🔍 Searching for image directories...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for root, dirs, files in os.walk(self.base_dir):
            image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
            if image_files:
                rel_path = Path(root).relative_to(self.base_dir)
                print(f"  📸 {rel_path}: {len(image_files)} images")

        print("-" * 50)

    def setup_complete_dataset(self, force_download: bool = False) -> Tuple[bool, str]:
        """
        Complete dataset setup pipeline with debugging
        """
        print("🌱 Plant Disease Dataset Setup")
        print("=" * 50)

        # Step 1: Download dataset
        if not self.download_dataset(force_download):
            print("\n🔍 Download failed, let's debug...")
            self.debug_dataset_structure()
            return False, "Failed to download dataset"

        # Step 2: Debug structure before organization
        self.debug_dataset_structure()

        # Step 3: Organize dataset
        if not self.organize_dataset():
            print("\n🔍 Organization failed, final structure:")
            self.debug_dataset_structure()
            return False, "Failed to organize dataset"

        # Step 4: Get dataset info
        info = self.get_dataset_info()

        print("\n📊 Final Dataset Information:")
        print(f"Path: {info['dataset_path']}")
        print(f"Classes: {info['classes']}")
        print(f"Total Images: {info['total_images']}")
        print("\nClass Distribution:")
        for class_name, count in info['class_distribution'].items():
            print(f"  {class_name}: {count} images")

        # Clean up zip file if it exists
        zip_files = list(self.base_dir.glob("*.zip"))
        for zip_file in zip_files:
            zip_file.unlink()
            print(f"\n🗑️  Cleaned up: {zip_file.name}")

        if info['total_images'] > 0:
            print("\n✅ Dataset setup completed successfully!")
            return True, str(self.dataset_dir)
        else:
            print("\n❌ No images found in dataset")
            return False, "No images found"


class AlternativeDatasetDownloader:
    """
    Alternative methods for dataset acquisition if Kaggle API fails
    """

    @staticmethod
    def download_sample_images():
        """
        Download sample plant disease images for testing
        """
        print("📥 Downloading sample plant disease images...")

        # Create sample dataset structure
        base_dir = Path("datasets/sample-plant-disease")
        base_dir.mkdir(parents=True, exist_ok=True)

        classes = ['Healthy', 'Powdery', 'Rust']

        # Sample URLs (in real implementation, use actual plant disease images)
        sample_urls = {
            'Healthy': [
                'https://example.com/healthy1.jpg',  # Replace with actual URLs
                'https://example.com/healthy2.jpg',
            ],
            'Powdery': [
                'https://example.com/powdery1.jpg',
                'https://example.com/powdery2.jpg',
            ],
            'Rust': [
                'https://example.com/rust1.jpg',
                'https://example.com/rust2.jpg',
            ]
        }

        for class_name in classes:
            class_dir = base_dir / class_name
            class_dir.mkdir(exist_ok=True)
            print(f"✅ Created directory: {class_name}")

        print("⚠️  Sample dataset structure created")
        print("📝 Please manually add plant disease images to each folder")
        print(f"Dataset path: {base_dir}")

        return str(base_dir)

    @staticmethod
    def create_demo_dataset():
        """
        Create a synthetic dataset for demonstration purposes
        """
        import numpy as np
        from PIL import Image

        print("🎭 Creating synthetic demo dataset...")

        base_dir = Path("datasets/demo-plant-disease")
        base_dir.mkdir(parents=True, exist_ok=True)

        classes = ['Healthy', 'Powdery', 'Rust']
        images_per_class = 100

        for class_name in classes:
            class_dir = base_dir / class_name
            class_dir.mkdir(exist_ok=True)

            for i in range(images_per_class):
                # Create synthetic image
                if class_name == 'Healthy':
                    # Green-ish images
                    img_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
                    img_array[:, :, 1] += 50  # More green
                elif class_name == 'Powdery':
                    # White-ish spots
                    img_array = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
                    img_array[50:150, 50:150] = 255  # White patches
                else:  # Rust
                    # Orange/brown-ish
                    img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                    img_array[:, :, 0] += 50  # More red
                    img_array[:, :, 1] += 30  # Some green

                # Save image
                img = Image.fromarray(img_array)
                img.save(class_dir / f"{class_name.lower()}_{i:03d}.jpg")

            print(f"✅ Created {images_per_class} synthetic {class_name} images")

        print(f"✅ Demo dataset created at: {base_dir}")
        return str(base_dir)


# Utility functions for easy usage
def setup_plant_disease_dataset(force_download: bool = False) -> Tuple[bool, str]:
    """
    Easy setup function for plant disease dataset
    """
    manager = KaggleDatasetManager()
    return manager.setup_complete_dataset(force_download)


def get_alternative_dataset() -> str:
    """
    Get alternative dataset if Kaggle fails
    """
    print("🔄 Kaggle dataset unavailable, creating alternative...")

    # Try sample dataset first
    try:
        return AlternativeDatasetDownloader.create_demo_dataset()
    except Exception as e:
        print(f"❌ Error creating demo dataset: {e}")
        return ""


if __name__ == "__main__":
    # Test the dataset manager
    success, dataset_path = setup_plant_disease_dataset()

    if success:
        print(f"\n🎉 Dataset ready at: {dataset_path}")
    else:
        print("\n⚠️  Using alternative dataset...")
        alt_path = get_alternative_dataset()
        if alt_path:
            print(f"📁 Alternative dataset at: {alt_path}")
        else:
            print("❌ Failed to setup any dataset")