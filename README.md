# GFormer with Advanced Checkpointing

A Graph Transformer for Recommendation Systems with comprehensive checkpointing and resume functionality.

## 🚀 New Features

### Advanced Checkpointing System
- **Automatic checkpoint saving** during training
- **Resume training** from any checkpoint
- **Best model tracking** based on performance metrics
- **Lightweight model weights** for inference
- **Kaggle-friendly** persistent training

## 📋 Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Checkpointing Guide](#checkpointing-guide)
5. [Resume Training](#resume-training)
6. [Kaggle Usage](#kaggle-usage)
7. [Parameters](#parameters)

## 🛠️ Installation

```bash
git clone https://github.com/your-username/GFormer.git
cd GFormer/GFormerAD/GFormer-main
pip install -r requirements.txt
```

## 📊 Data Preparation

### Using Your Own Dataset (e.g., Declic_Augmented)

1. **Prepare your data** using the data processing notebook
2. **Place dataset files** in the correct directory:
```
GFormerAD/Datasets/Declic_Augmented/
├── trnMat.pkl    # Training matrix
├── tstMat.pkl    # Test matrix
├── valMat.pkl    # Validation matrix
└── mappings.pkl  # User/item mappings
```

## 🎯 Training

### Basic Training (From Scratch)
```bash
python Main.py --data Declic_Augmented --epoch 100
```

### Training with Custom Parameters
```bash
python Main.py --data Declic_Augmented \
               --epoch 100 \
               --lr 0.001 \
               --batch 4096 \
               --latdim 64 \
               --save_path my_model
```

## 💾 Checkpointing Guide

### What Gets Saved

The checkpointing system saves:
- ✅ **Model weights** (complete state)
- ✅ **Optimizer state** (learning rate, momentum, etc.)
- ✅ **Training progress** (current epoch)
- ✅ **Best metrics** (best recall, NDCG)
- ✅ **Training history** (loss curves, performance)
- ✅ **Hyperparameters** (all configuration)

### File Structure After Training

```
GFormer-main/
├── Checkpoints/
│   ├── checkpoint_epoch_10.pth      # Regular checkpoints
│   ├── checkpoint_epoch_20.pth
│   ├── checkpoint_epoch_30.pth
│   ├── latest_checkpoint.pth        # Always most recent
│   └── final_checkpoint.pth         # Final model
├── BestCheckpoints/
│   └── best_checkpoint_epoch_25.pth # Best performing model
├── Models/
│   ├── weights_epoch_20.pth         # Lightweight weights
│   └── weights_epoch_40.pth
└── History/
    └── training_metrics.his         # Training history
```

### Checkpoint Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save_freq` | 10 | Save checkpoint every N epochs |
| `--keep_checkpoints` | 5 | Number of regular checkpoints to keep |
| `--save_weights_freq` | 20 | Save model weights every N epochs |
| `--auto_save` | True | Enable automatic checkpoint saving |
| `--save_best_only` | False | Only save when performance improves |

## 🔄 Resume Training

### Resume from Latest Checkpoint
```bash
python Main.py --data Declic_Augmented --epoch 100 --resume
```

### Resume from Specific Checkpoint
```bash
python Main.py --data Declic_Augmented \
               --epoch 100 \
               --load_checkpoint Checkpoints/checkpoint_epoch_30.pth
```

### Resume from Best Checkpoint
```bash
python Main.py --data Declic_Augmented --epoch 100 --load_best
```

### Load Only Model Weights (for Inference)
```bash
python Main.py --data Declic_Augmented \
               --load_checkpoint Models/weights_epoch_50.pth \
               --epoch 0  # No training, just evaluation
```

## 🏆 Kaggle Usage

Perfect for Kaggle's time-limited environment!

### Initial Training Session
```bash
python Main.py --data Declic_Augmented \
               --epoch 100 \
               --save_freq 5 \
               --keep_checkpoints 10
```

### When Kaggle Session Ends
1. **Download checkpoint files** from the output
2. **Start new Kaggle session**
3. **Upload checkpoint files** to the new session
4. **Resume training:**

```bash
python Main.py --data Declic_Augmented --epoch 100 --resume
```

### Kaggle Pro Tips
- Set `--save_freq 5` for frequent saves
- Set `--keep_checkpoints 10` to keep more backups
- Always download both `latest_checkpoint.pth` and `best_checkpoint_*.pth`
- Use `--load_best` to resume from the best performing model

## 🎛️ Parameters

### Core Training Parameters
```bash
--data          # Dataset name (e.g., 'Declic_Augmented')
--epoch         # Number of training epochs (default: 100)
--lr            # Learning rate (default: 0.001)
--batch         # Batch size (default: 4096)
--latdim        # Embedding dimension (default: 32)
--gcn_layer     # Number of GCN layers (default: 2)
--gt_layer      # Number of Graph Transformer layers (default: 1)
--save_path     # Model save name (default: 'tem')
```

### Checkpointing Parameters
```bash
--resume                # Resume from latest checkpoint
--load_checkpoint PATH  # Load specific checkpoint file
--load_best            # Load best checkpoint instead of latest
--save_freq N          # Save checkpoint every N epochs (default: 10)
--keep_checkpoints N   # Keep N regular checkpoints (default: 5)
--save_weights_freq N  # Save weights every N epochs (default: 20)
--auto_save            # Enable automatic checkpointing (default: True)
--save_best_only       # Only save when performance improves
```

### Advanced Parameters
```bash
--ssl_reg       # Contrastive regularization (default: 1)
--reg           # Weight decay regularization (default: 1e-4)
--topk          # Top-K for evaluation (default: 40)
--tstEpoch      # Test every N epochs (default: 1)
--gpu           # GPU ID to use (default: '0')
```

## 📈 Example Training Workflows

### 1. Quick Experimentation
```bash
python Main.py --data Declic_Augmented --epoch 50 --save_freq 10
```

### 2. Long Training with Frequent Saves
```bash
python Main.py --data Declic_Augmented \
               --epoch 200 \
               --save_freq 5 \
               --keep_checkpoints 20
```

### 3. Resume Long Training
```bash
python Main.py --data Declic_Augmented \
               --epoch 200 \
               --resume
```

### 4. Fine-tuning from Best Model
```bash
python Main.py --data Declic_Augmented \
               --epoch 50 \
               --load_best \
               --lr 0.0001  # Lower learning rate for fine-tuning
```

## 🔧 Troubleshooting

### Common Issues

**1. Checkpoint not found**
```bash
# Check if checkpoint exists
ls Checkpoints/
# Use absolute path if needed
--load_checkpoint /full/path/to/checkpoint.pth
```

**2. CUDA out of memory**
```bash
# Reduce batch size
--batch 2048
# Or reduce embedding dimension
--latdim 16
```

**3. Resume training from wrong epoch**
```bash
# Check checkpoint epoch
python -c "import torch; print(torch.load('Checkpoints/latest_checkpoint.pth')['epoch'])"
```

### File Size Management

**Large checkpoint files?**
```bash
# Use save_best_only to reduce storage
--save_best_only
# Or reduce checkpoint frequency
--save_freq 20
```

## 🏁 Getting Started

1. **Clone the repository**
2. **Prepare your dataset** (see Data Preparation)
3. **Start training:**
   ```bash
   python Main.py --data Declic_Augmented --epoch 100
   ```
4. **Monitor training** and checkpoints will be saved automatically
5. **Resume anytime** with `--resume` if interrupted

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the parameter descriptions

---

**Happy Training! 🚀**
