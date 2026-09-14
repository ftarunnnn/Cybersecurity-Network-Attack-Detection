import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_generator import NetworkDatasetGenerator

def main():
    print("[Phase 2] Generating Synthetic Network Traffic Datasets (CIC-IDS2017 / UNSW-NB15 Schema)...")
    
    raw_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    generator = NetworkDatasetGenerator(seed=42)
    
    # Train dataset: 6,000 samples
    train_df = generator.generate_dataset(num_samples=6000)
    train_path = os.path.join(raw_dir, 'network_traffic.csv')
    train_df.to_csv(train_path, index=False)
    print(f"[OK] Training dataset saved to: {train_path} ({len(train_df)} rows, {len(train_df.columns)} columns)")

    # Test dataset: 1,500 samples
    test_generator = NetworkDatasetGenerator(seed=123)
    test_df = test_generator.generate_dataset(num_samples=1500)
    test_path = os.path.join(raw_dir, 'network_traffic_test.csv')
    test_df.to_csv(test_path, index=False)
    print(f"[OK] Test dataset saved to: {test_path} ({len(test_df)} rows, {len(test_df.columns)} columns)")

    print("\nClass distribution in generated training dataset:")
    print(train_df['label'].value_counts())

if __name__ == '__main__':
    main()
