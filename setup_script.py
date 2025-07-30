#!/usr/bin/env python3
"""
Quick fix script to reorganize the dataset properly and fix the Keras issue
"""

import shutil
from pathlib import Path


def fix_dataset_organization():
    """
    Fix the dataset to use the larger Train directory
    """
    print("🔧 Fixing dataset organization...")

    base_dir = Path("datasets")
    dataset_dir = base_dir / "plant-disease-recognition"
    train_dir = base_dir / "Train" / "Train"
    validation_dir = base_dir / "Validation" / "Validation"

    if not train_dir.exists():
        print("❌ Train directory not found")
        return False

    print(f"📁 Using larger dataset from: {train_dir}")

    # Remove old small dataset
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Create new dataset directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy all classes from Train and Validation
    classes = ['Healthy', 'Powdery', 'Rust']

    for class_name in classes:
        target_class_dir = dataset_dir / class_name
        target_class_dir.mkdir(exist_ok=True)

        # Copy training images
        train_class_dir = train_dir / class_name
        if train_class_dir.exists():
            for img in train_class_dir.glob("*"):
                if img.is_file():
                    shutil.copy2(img, target_class_dir)

        # Copy validation images
        val_class_dir = validation_dir / class_name
        if val_class_dir.exists():
            for img in val_class_dir.glob("*"):
                if img.is_file():
                    new_name = f"val_{img.name}"
                    shutil.copy2(img, target_class_dir / new_name)

        total_images = len(list(target_class_dir.glob("*")))
        print(f"✅ {class_name}: {total_images} images")

    print("✅ Dataset reorganization completed!")
    return True


def check_tensorflow_version():
    """
    Check TensorFlow version and provide fix suggestions
    """
    try:
        import tensorflow as tf
        print(f"📦 TensorFlow version: {tf.__version__}")

        # Test the Rescaling layer
        try:
            rescaling = tf.keras.layers.Rescaling(1. / 255)
            print("✅ Rescaling layer works correctly")
        except AttributeError:
            print("❌ Rescaling layer issue found")
            print("💡 Fix: Use tf.keras.layers.Rescaling instead of tf.keras.utils.Rescaling")

    except ImportError:
        print("❌ TensorFlow not installed")


def main():
    """
    Main fix function
    """
    print("🛠️  Quick Fix for Plant Disease Classification")
    print("=" * 50)

    # Fix 1: Dataset organization
    fix_dataset_organization()

    # Fix 2: Check TensorFlow
    print("\n🔍 Checking TensorFlow configuration...")
    check_tensorflow_version()

    print("\n🎉 Fixes applied!")
    print("\nNow run: python plant_disease_classifier.py")


if __name__ == "__main__":
    main()