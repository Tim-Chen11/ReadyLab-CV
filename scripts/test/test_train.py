#!/usr/bin/env python
"""
Debug script to check dataset output types
Save this as: scripts/debug_dataset.py
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.data.data_utils import create_data_loaders

def debug_dataset():
    """Debug dataset to find the data type issue"""
    
    print("🏗️ Project structure debug")
    print(f"📁 Script location: {Path(__file__).parent}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Data directory: {project_root / 'data'}")
    
    # Check if data files exist
    data_dir = project_root / 'data'
    train_split = data_dir / 'splits' / 'train.json'
    val_split = data_dir / 'splits' / 'val.json'
    
    print(f"\n📋 File existence check:")
    print(f"  Train split: {train_split.exists()} - {train_split}")
    print(f"  Val split: {val_split.exists()} - {val_split}")
    
    if not train_split.exists():
        print("❌ Train split file not found!")
        return
    
    # Create minimal config for debugging
    config = {
        'data_dir': str(data_dir),  # Use absolute path
        'batch_size': 4,  # Small batch for debugging
        'num_workers': 0,  # No multiprocessing for easier debugging
        'use_cached': False,
        'model_name': 'resnet50',
        'num_classes': 5
    }
    
    print(f"\n🔍 Creating data loaders with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        train_loader, val_loader, class_weights, class_names = create_data_loaders(config)
        print(f"✅ Data loaders created successfully")
        print(f"📊 Class names: {class_names}")
        print(f"📊 Train dataset size: {len(train_loader.dataset)}")
        print(f"📊 Val dataset size: {len(val_loader.dataset)}")
        
        # Check first batch
        print(f"\n🔬 Checking first batch...")
        for i, batch in enumerate(train_loader):
            print(f"\nBatch {i}:")
            print(f"  Type of batch: {type(batch)}")
            
            if isinstance(batch, (list, tuple)):
                print(f"  Batch length: {len(batch)}")
                for j, item in enumerate(batch):
                    print(f"    Item {j}: type={type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"             shape={item.shape}")
                    if hasattr(item, 'dtype'):
                        print(f"             dtype={item.dtype}")
                    
                    # If it's supposed to be a tensor but isn't, show content
                    if not isinstance(item, torch.Tensor):
                        try:
                            if hasattr(item, '__iter__') and not isinstance(item, str):
                                content = list(item)[:5] if len(item) > 5 else list(item)
                                print(f"             content preview: {content}")
                            else:
                                print(f"             content: {item}")
                        except Exception as e:
                            print(f"             content preview error: {e}")
            else:
                print(f"  Batch content: {batch}")
            
            # Only check first batch
            break
            
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to load just the JSON to see the data format
        print(f"\n🔍 Checking raw JSON data...")
        try:
            import json
            with open(train_split, 'r') as f:
                data = json.load(f)
            print(f"📄 JSON loaded successfully: {len(data)} items")
            if data:
                print(f"📋 First item keys: {list(data[0].keys())}")
                print(f"📋 First item: {data[0]}")
        except Exception as json_e:
            print(f"❌ Error reading JSON: {json_e}")

if __name__ == '__main__':
    debug_dataset()