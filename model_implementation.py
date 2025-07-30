# Simplified Plant Disease Classifier - Working Version
# Fixes all the issues and provides reliable results

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class SimplePlantDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = []
        self.img_height = 224
        self.img_width = 224

    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the plant disease dataset"""
        print("Loading and preprocessing plant disease data...")

        # Create datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32
        )

        # Get class names
        self.class_names = train_ds.class_names
        print(f"Disease classes found: {self.class_names}")

        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds

    def build_model(self, num_classes):
        """Build optimized model for plant disease classification"""
        print("Building optimized model...")

        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1)
        ])

        # Base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze initially

        # Build model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs, outputs)

        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def train_model(self, train_ds, val_ds, epochs=25):
        """Train the model with proper callbacks"""
        print("Training model...")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def fine_tune_model(self, train_ds, val_ds, epochs=10):
        """Fine-tune with unfrozen layers"""
        print("Fine-tuning model...")

        # Unfreeze some layers
        self.model.layers[3].trainable = True  # Unfreeze base model

        # Freeze early layers
        for layer in self.model.layers[3].layers[:-20]:
            layer.trainable = False

        # Compile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Continue training
        initial_epochs = len(self.history.history['loss'])

        history_fine = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs + epochs,
            initial_epoch=initial_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        # Safely extend history
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            if key in history_fine.history:
                self.history.history[key].extend(history_fine.history[key])

    def evaluate_model(self, test_ds):
        """Evaluate model and return metrics"""
        print("Evaluating model...")

        predictions = self.model.predict(test_ds)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get true labels
        true_labels = []
        for _, labels in test_ds:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)

        # Calculate accuracy
        test_accuracy = np.mean(predicted_classes == true_labels)

        return test_accuracy, predicted_classes, true_labels

    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, true_labels, predicted_classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predicted_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Plant Disease Classification - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return cm

    def generate_classification_report(self, true_labels, predicted_classes):
        """Generate detailed classification report"""
        report = classification_report(
            true_labels,
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )

        print("\n📊 Classification Report:")
        print("=" * 50)
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1-score']:.3f}")
            print()

        print(f"Overall Accuracy: {report['accuracy']:.3f}")
        print(f"Macro Avg F1:     {report['macro avg']['f1-score']:.3f}")

    def save_model(self, model_path="plant_disease_model.keras", save_weights_only=False):
        """Save the trained model"""
        try:
            if save_weights_only:
                self.model.save_weights(model_path.replace('.keras', '_weights.weights.h5'))
                print(f"💾 Model weights saved to: {model_path.replace('.keras', '_weights.weights.h5')}")
            else:
                self.model.save(model_path)
                print(f"💾 Complete model saved to: {model_path}")

            # Also save model information
            import json
            model_info = {
                'class_names': self.class_names,
                'img_height': self.img_height,
                'img_width': self.img_width,
                'test_accuracy': getattr(self, 'test_accuracy', None)
            }

            info_path = model_path.replace('.keras', '_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"📋 Model info saved to: {info_path}")

            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_saved_model(self, model_path="plant_disease_model.keras"):
        """Load a previously saved model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")

            # Load model info if available
            info_path = model_path.replace('.keras', '_info.json')
            if Path(info_path).exists():
                import json
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                self.class_names = model_info.get('class_names', self.class_names)
                print(f"📋 Model info loaded: {len(self.class_names)} classes")

            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def main():
    """Main function - simplified and reliable"""
    print("🌱 Simplified Plant Disease Classification")
    print("=" * 50)

    # Check if dataset exists
    dataset_path = "datasets/plant-disease-recognition"
    if not Path(dataset_path).exists():
        print("❌ Dataset not found!")
        print(f"Expected location: {dataset_path}")
        print("Please run the dataset setup first.")
        return

    # Initialize classifier
    classifier = SimplePlantDiseaseClassifier()

    # Load data
    try:
        train_ds, val_ds = classifier.load_and_preprocess_data(dataset_path)

        # Create test set from validation
        val_batches = tf.data.experimental.cardinality(val_ds)
        test_ds = val_ds.take(val_batches // 3)
        val_ds = val_ds.skip(val_batches // 3)

        num_classes = len(classifier.class_names)
        print(f"✅ Dataset loaded: {num_classes} classes")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Build model
    model = classifier.build_model(num_classes)
    print(f"✅ Model built with {model.count_params():,} parameters")

    # Train model (single phase for reliability)
    print("\n🚀 Training model...")
    history = classifier.train_model(train_ds, val_ds, epochs=20)

    # Optional fine-tuning (comment out if having issues)
    try:
        print("\n🔧 Fine-tuning model...")
        classifier.fine_tune_model(train_ds, val_ds, epochs=8)
    except Exception as e:
        print(f"⚠️  Fine-tuning skipped due to error: {e}")

    # Evaluate
    print("\n📊 Evaluating model...")
    test_accuracy, predicted_classes, true_labels = classifier.evaluate_model(test_ds)

    print(f"\n✅ Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Generate visualizations
    print("\n📈 Generating results...")

    # Training history
    classifier.plot_training_history()

    # Confusion matrix
    cm = classifier.plot_confusion_matrix(true_labels, predicted_classes)

    # Classification report
    report = classifier.generate_classification_report(true_labels, predicted_classes)

    # ACTUALLY SAVE THE MODEL HERE!
    print("\n💾 Saving trained model...")
    classifier.test_accuracy = test_accuracy
    classifier.save_model("plant_disease_model.keras")

    # Agricultural impact
    print("\n🌾 Agricultural Impact Assessment:")
    print("-" * 40)
    if test_accuracy > 0.85:
        print("✅ Model suitable for agricultural deployment")
        print(f"   Disease detection rate: {test_accuracy * 100:.1f}%")
        print("   Recommended for field trials")
    elif test_accuracy > 0.75:
        print("⚠️  Model shows promise but needs improvement")
        print("   Consider more training data or architecture changes")
    else:
        print("❌ Model requires significant improvement")
        print("   Not ready for agricultural deployment")

    print(f"\n🎉 Plant Disease Classification Completed!")
    print(f"Final Performance: {test_accuracy * 100:.2f}% accuracy")
    print(f"💾 Model saved to: plant_disease_model.keras")


if __name__ == "__main__":
    main()