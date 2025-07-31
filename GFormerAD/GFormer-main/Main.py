import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph, GTLayer, ResidualGTLayer
from DataHandler import DataHandler
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch as t


class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.distill_weight = 0.1
        self.ResidualGTLayer = ResidualGTLayer()
        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        
        # Checkpointing setup
        self.checkpoint_dir = 'Checkpoints'
        self.best_checkpoint_dir = 'BestCheckpoints'
        self.create_checkpoint_dirs()
        self.best_recall = 0.0
        self.best_ndcg = 0.0
        self.start_epoch = 0
        self.frozen_layers = set()  # Track frozen layers
        self.layer_freeze_history = []  # Track freezing history

    def get_ordered_parameters(self):
        """Get model parameters in a logical order for percentage-based freezing"""
        ordered_params = []
        
        # Order: Embeddings -> GCN -> GT -> PNN
        # 1. Embeddings (usually frozen first in fine-tuning)
        ordered_params.extend([
            ('uEmbeds', self.model.uEmbeds),
            ('iEmbeds', self.model.iEmbeds)
        ])
        
        # 2. GCN Layers (backbone features)
        for i, layer in enumerate(self.model.gcnLayers):
            for name, param in layer.named_parameters():
                ordered_params.append((f'gcnLayers.{i}.{name}', param))
        
        # 3. GT Layers (attention mechanism)
        for name, param in self.model.gtLayers.named_parameters():
            ordered_params.append((f'gtLayers.{name}', param))
        
        # 4. PNN Layers (final prediction layers - usually kept trainable)
        for i, layer in enumerate(self.model.pnnLayers):
            for name, param in layer.named_parameters():
                ordered_params.append((f'pnnLayers.{i}.{name}', param))
        
        return ordered_params

    def freeze_first_percent(self, percent):
        """Freeze the first X% of layers (typically lower-level features)"""
        if percent <= 0:
            return 0
            
        ordered_params = self.get_ordered_parameters()
        total_layers = len(ordered_params)
        freeze_count = int(total_layers * percent)
        
        frozen_count = 0
        log(f"🧊 Freezing first {percent*100:.1f}% of layers ({freeze_count}/{total_layers} layers)")
        
        for i in range(freeze_count):
            if i < len(ordered_params):
                name, param = ordered_params[i]
                param.requires_grad = False
                self.frozen_layers.add(name)
                frozen_count += 1
                log(f"   ❄️  Frozen: {name}")
        
        return frozen_count

    def freeze_last_percent(self, percent):
        """Freeze the last X% of layers (typically higher-level features)"""
        if percent <= 0:
            return 0
            
        ordered_params = self.get_ordered_parameters()
        total_layers = len(ordered_params)
        freeze_count = int(total_layers * percent)
        
        frozen_count = 0
        log(f"🧊 Freezing last {percent*100:.1f}% of layers ({freeze_count}/{total_layers} layers)")
        
        start_idx = total_layers - freeze_count
        for i in range(start_idx, total_layers):
            if i < len(ordered_params):
                name, param = ordered_params[i]
                param.requires_grad = False
                self.frozen_layers.add(name)
                frozen_count += 1
                log(f"   ❄️  Frozen: {name}")
        
        return frozen_count

    def freeze_backbone_keep_head(self):
        """Freeze backbone (embeddings + GCN + GT), keep PNN trainable"""
        frozen_count = 0
        log("🧊 Freezing backbone layers (Embeddings + GCN + GT), keeping PNN trainable")
        
        # Freeze embeddings
        self.model.uEmbeds.requires_grad = False
        self.model.iEmbeds.requires_grad = False
        self.frozen_layers.add('uEmbeds')
        self.frozen_layers.add('iEmbeds')
        frozen_count += 2
        
        # Freeze GCN layers
        for name, param in self.model.gcnLayers.named_parameters():
            param.requires_grad = False
            full_name = f'gcnLayers.{name}'
            self.frozen_layers.add(full_name)
            frozen_count += 1
            log(f"   ❄️  Frozen: {full_name}")
        
        # Freeze GT layers
        for name, param in self.model.gtLayers.named_parameters():
            param.requires_grad = False
            full_name = f'gtLayers.{name}'
            self.frozen_layers.add(full_name)
            frozen_count += 1
            log(f"   ❄️  Frozen: {full_name}")
        
        log(f"   ✅ PNN layers kept trainable for task-specific adaptation")
        return frozen_count

    def progressive_unfreeze_layers(self, current_epoch, total_epochs):
        """Progressively unfreeze layers during training"""
        if not args.progressive_unfreeze:
            return
        
        # Calculate unfreezing progress
        progress = current_epoch / total_epochs
        
        if args.unfreeze_schedule == 'linear':
            unfreeze_ratio = progress
        elif args.unfreeze_schedule == 'exponential':
            unfreeze_ratio = progress ** 2
        else:
            unfreeze_ratio = progress
        
        # Determine which layers to unfreeze
        ordered_params = self.get_ordered_parameters()
        total_frozen = len(self.frozen_layers)
        
        if total_frozen == 0:
            return
        
        target_unfrozen = int(total_frozen * unfreeze_ratio)
        current_unfrozen = 0
        
        # Unfreeze layers in reverse order (unfreeze higher-level features first)
        unfrozen_this_step = []
        for name, param in reversed(ordered_params):
            if name in self.frozen_layers and current_unfrozen < target_unfrozen:
                param.requires_grad = True
                self.frozen_layers.remove(name)
                unfrozen_this_step.append(name)
                current_unfrozen += 1
        
        if unfrozen_this_step:
            log(f"🔓 Progressive unfreezing at epoch {current_epoch} ({progress*100:.1f}% progress):")
            for name in unfrozen_this_step:
                log(f"   🔥 Unfrozen: {name}")
            
            # Update optimizer with new trainable parameters
            self.update_optimizer_for_unfrozen_layers()

    def update_optimizer_for_unfrozen_layers(self):
        """Update optimizer when layers are unfrozen during training"""
        # Get current trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create new optimizer with potentially different learning rate for newly unfrozen layers
        if args.fine_tune_lr is not None:
            lr = args.fine_tune_lr
        else:
            lr = args.lr
        
        # Scale learning rate for newly unfrozen layers
        if hasattr(args, 'frozen_lr_scale'):
            lr_scaled = lr * args.frozen_lr_scale
        else:
            lr_scaled = lr
        
        self.opt = torch.optim.Adam(trainable_params, lr=lr_scaled, weight_decay=0)
        log(f"🔄 Optimizer updated with LR={lr_scaled:.6f} for newly unfrozen layers")

    def apply_freezing_strategy(self):
        """Apply the specified freezing strategy"""
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_count = 0
        
        log("=" * 60)
        log("🧊 APPLYING FREEZING STRATEGY")
        log("=" * 60)
        
        # Strategy 1: Freeze first X% of layers
        if args.freeze_first_percent > 0:
            frozen_count += self.freeze_first_percent(args.freeze_first_percent)
        
        # Strategy 2: Freeze last X% of layers
        if args.freeze_last_percent > 0:
            frozen_count += self.freeze_last_percent(args.freeze_last_percent)
        
        # Strategy 3: Freeze embeddings only
        if args.freeze_embeddings:
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.requires_grad = False
            self.frozen_layers.add('uEmbeds')
            self.frozen_layers.add('iEmbeds')
            frozen_count += 2
            log("❄️  Embeddings frozen")
        
        # Strategy 4: Freeze backbone, keep head
        if args.freeze_backbone:
            frozen_count += self.freeze_backbone_keep_head()
        
        # Calculate statistics
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        log("=" * 60)
        log("📊 FREEZING SUMMARY")
        log("=" * 60)
        log(f"Total parameters: {total_params:,}")
        log(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        log(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        log(f"Frozen layers: {len(self.frozen_layers)}")
        
        if hasattr(args, 'progressive_unfreeze') and args.progressive_unfreeze:
            log(f"🔄 Progressive unfreezing enabled ({getattr(args, 'unfreeze_schedule', 'linear')} schedule)")
        
        # Log which components are trainable
        self.log_component_status()
        
        return trainable_params, frozen_params

    def log_component_status(self):
        """Log the status of each component"""
        log("🔍 COMPONENT STATUS:")
        
        # Check embeddings
        if self.model.uEmbeds.requires_grad or self.model.iEmbeds.requires_grad:
            log("   ✅ Embeddings: Trainable")
        else:
            log("   ❄️  Embeddings: Frozen")
        
        # Check GCN layers
        gcn_trainable = any(p.requires_grad for p in self.model.gcnLayers.parameters())
        if gcn_trainable:
            trainable_gcn = sum(1 for p in self.model.gcnLayers.parameters() if p.requires_grad)
            total_gcn = sum(1 for p in self.model.gcnLayers.parameters())
            log(f"   ✅ GCN Layers: {trainable_gcn}/{total_gcn} trainable")
        else:
            log("   ❄️  GCN Layers: Frozen")
        
        # Check GT layers
        gt_trainable = any(p.requires_grad for p in self.model.gtLayers.parameters())
        if gt_trainable:
            trainable_gt = sum(1 for p in self.model.gtLayers.parameters() if p.requires_grad)
            total_gt = sum(1 for p in self.model.gtLayers.parameters())
            log(f"   ✅ GT Layers: {trainable_gt}/{total_gt} trainable")
        else:
            log("   ❄️  GT Layers: Frozen")
        
        # Check PNN layers
        pnn_trainable = any(p.requires_grad for p in self.model.pnnLayers.parameters())
        if pnn_trainable:
            trainable_pnn = sum(1 for p in self.model.pnnLayers.parameters() if p.requires_grad)
            total_pnn = sum(1 for p in self.model.pnnLayers.parameters())
            log(f"   ✅ PNN Layers: {trainable_pnn}/{total_pnn} trainable")
        else:
            log("   ❄️  PNN Layers: Frozen")

    def setup_fine_tuning_optimizer(self):
        """Setup optimizer for fine-tuning with appropriate learning rates"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if hasattr(args, 'fine_tune_lr') and args.fine_tune_lr is not None:
            lr = args.fine_tune_lr
            log(f"📈 Using fine-tuning learning rate: {lr}")
        else:
            lr = args.lr
            log(f"📈 Using standard learning rate: {lr}")
        
        self.opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=0)
        
        log(f"✅ Optimizer created with {len(trainable_params):,} trainable parameters")

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.ResidualGTLayer).cuda()
        self.distill_model = Model(self.ResidualGTLayer).cuda()
        
        # Apply freezing strategy BEFORE creating optimizer
        if (hasattr(args, 'freeze_first_percent') and args.freeze_first_percent > 0) or \
           (hasattr(args, 'freeze_last_percent') and args.freeze_last_percent > 0) or \
           (hasattr(args, 'freeze_embeddings') and args.freeze_embeddings) or \
           (hasattr(args, 'freeze_backbone') and args.freeze_backbone):
            self.apply_freezing_strategy()
        
        # Setup optimizer after freezing
        self.setup_fine_tuning_optimizer()
        
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def create_checkpoint_dirs(self):
        """Create checkpoint directories if they don't exist"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_checkpoint_dir, exist_ok=True)
        os.makedirs('Models', exist_ok=True)
        os.makedirs('History', exist_ok=True)

    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """Save comprehensive checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'distill_model_state_dict': self.distill_model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'gtLayer_state_dict': self.gtLayer.state_dict(),
            'metrics': self.metrics,
            'best_recall': self.best_recall,
            'best_ndcg': self.best_ndcg,
            'args': vars(args),
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        # Save regular checkpoint
        if is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'final_checkpoint.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        log(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = os.path.join(self.best_checkpoint_dir, f'best_checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, best_checkpoint_path)
            log(f'Best checkpoint saved: {best_checkpoint_path}')
        
        # Save latest checkpoint (always overwrite)
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Keep only last N checkpoints to save space
        if hasattr(args, 'keep_checkpoints'):
            self.cleanup_old_checkpoints(keep_last=args.keep_checkpoints)
        else:
            self.cleanup_old_checkpoints(keep_last=5)

    def cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoints to save disk space"""
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
            
            if len(checkpoint_files) > keep_last:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                
                # Remove oldest checkpoints
                for old_file in checkpoint_files[:-keep_last]:
                    old_path = os.path.join(self.checkpoint_dir, old_file)
                    os.remove(old_path)
                    log(f'Removed old checkpoint: {old_file}')
        except Exception as e:
            log(f'Error cleaning up checkpoints: {e}')

    def load_checkpoint(self, checkpoint_path=None, load_best=False):
        """Load checkpoint from file - UPDATED FOR EVALUATION MODE"""
        if checkpoint_path is None:
            if load_best:
                # Find the best checkpoint
                if not os.path.exists(self.best_checkpoint_dir):
                    log('Best checkpoint directory does not exist')
                    return False
                    
                best_files = [f for f in os.listdir(self.best_checkpoint_dir) 
                            if f.startswith('best_checkpoint_') and f.endswith('.pth')]
                if best_files:
                    best_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
                    checkpoint_path = os.path.join(self.best_checkpoint_dir, best_files[0])
                else:
                    log('No best checkpoint found')
                    return False
            else:
                # Load latest checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        
        if not os.path.exists(checkpoint_path):
            log(f'Checkpoint not found: {checkpoint_path}')
            return False
        
        try:
            # FIXED: Use weights_only=False to allow numpy objects and other pickle data
            checkpoint = torch.load(checkpoint_path, 
                                map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                weights_only=False)
            
            # Load model states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.distill_model.load_state_dict(checkpoint['distill_model_state_dict'])
            self.gtLayer.load_state_dict(checkpoint['gtLayer_state_dict'])
            
            # FIXED: Only load optimizer if we're training (epoch > 0)
            if args.epoch > 0:
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                log('Optimizer state loaded for training')
            else:
                log('Optimizer state skipped for evaluation-only mode')
            
            # Load training progress
            self.metrics = checkpoint['metrics']
            self.best_recall = checkpoint.get('best_recall', 0.0)
            self.best_ndcg = checkpoint.get('best_ndcg', 0.0)
            self.start_epoch = checkpoint['epoch'] + 1
            
            log(f'Checkpoint loaded: {checkpoint_path}')
            log(f'Original epoch: {checkpoint["epoch"]}')
            log(f'Best Recall in checkpoint: {self.best_recall:.4f}')
            log(f'Best NDCG in checkpoint: {self.best_ndcg:.4f}')
            
            # FIXED: Set appropriate mode based on whether we're training or evaluating
            if args.epoch == 0:
                self.model.eval()
                self.distill_model.eval()
                log('Models set to evaluation mode')
            else:
                self.model.train()
                self.distill_model.train()
                log('Models set to training mode')
            
            return True
            
        except Exception as e:
            log(f'Error loading checkpoint: {e}')
            return False

    def save_model_weights(self, epoch, suffix=''):
        """Save only model weights (lighter than full checkpoint)"""
        weights = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        weights_path = os.path.join('Models', f'weights_epoch_{epoch}{suffix}.pth')
        torch.save(weights, weights_path)
        log(f'Model weights saved: {weights_path}')

    def load_model_weights(self, weights_path):
        """Enhanced model weights loading for evaluation"""
        if not os.path.exists(weights_path):
            log(f'Weights file not found: {weights_path}')
            return False
        
        try:
            # Determine file extension
            file_ext = weights_path.split('.')[-1].lower()
            log(f'Loading weights file with extension: .{file_ext}')
            
            weights = torch.load(weights_path, 
                            map_location='cuda' if torch.cuda.is_available() else 'cpu',
                            weights_only=False)
            
            # Handle different weight file formats
            if file_ext == 'mod':
                # Handle .mod files (legacy GFormer format)
                if 'model' in weights:
                    # Method 1: Load the entire model object
                    log('Loading .mod file with model object')
                    loaded_model = weights['model']
                    
                    # Copy the parameters from loaded model to current model
                    self.model.load_state_dict(loaded_model.state_dict())
                    log('Model state dict loaded from .mod file')
                    
                else:
                    log('Invalid .mod file format - no model key found')
                    return False
                    
            else:
                # Handle .pth, .pt files
                if 'model_state_dict' in weights:
                    # Standard checkpoint format
                    self.model.load_state_dict(weights['model_state_dict'])
                    log('Model state dict loaded from checkpoint')
                    
                elif 'model' in weights:
                    # Handle nested model format
                    if hasattr(weights['model'], 'state_dict'):
                        self.model.load_state_dict(weights['model'].state_dict())
                        log('Model state dict loaded from nested model')
                    else:
                        self.model = weights['model']
                        log('Entire model object loaded')
                        
                else:
                    # Direct state dict
                    self.model.load_state_dict(weights)
                    log('Direct state dict loaded')
            
            # Sync distillation model
            self.distill_model.load_state_dict(self.model.state_dict())
            log('Distillation model synchronized with main model')
            
            # Recreate optimizer with loaded model parameters
            self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
            log('Optimizer recreated with loaded model parameters')
            
            log(f'Model weights loaded successfully: {weights_path}')
            return True
            
        except Exception as e:
            log(f'Error loading weights: {e}')
            log('Trying alternative loading method...')
            
            # Alternative method: Try loading as legacy format
            try:
                return self.load_legacy_model(weights_path)
            except Exception as e2:
                log(f'Alternative loading also failed: {e2}')
                return False

    def load_legacy_model(self, weights_path):
        """Load legacy .mod format models"""
        try:
            log('Attempting legacy model loading...')
            ckp = torch.load(weights_path, weights_only=False)
            
            if 'model' in ckp:
                # Replace current model with loaded model
                self.model = ckp['model']
                
                # Recreate distillation model with same architecture
                self.distill_model = Model(self.ResidualGTLayer).cuda()
                self.distill_model.load_state_dict(self.model.state_dict())
                
                # Recreate optimizer
                self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
                
                log('Legacy model loaded successfully')
                return True
            else:
                log('No model found in legacy file')
                return False
                
        except Exception as e:
            log(f'Legacy loading failed: {e}')
            return False

    def load_model_weights_for_transfer(self, weights_path):
        """FIXED: Load weights for transfer learning, handling dataset size mismatches"""
        if not os.path.exists(weights_path):
            log(f'Weights file not found: {weights_path}')
            return False
        
        try:
            log(f'🔄 Loading weights for transfer learning: {weights_path}')
            
            # Determine file extension
            file_ext = weights_path.split('.')[-1].lower()
            log(f'Loading weights file with extension: .{file_ext}')
            
            weights = torch.load(weights_path, 
                            map_location='cuda' if torch.cuda.is_available() else 'cpu',
                            weights_only=False)
            
            # Extract source model state
            if file_ext == 'mod' and 'model' in weights:
                source_model = weights['model']
                source_state = source_model.state_dict()
                log('✅ Extracted state dict from .mod file')
            elif 'model_state_dict' in weights:
                source_state = weights['model_state_dict']
                log('✅ Using model_state_dict from checkpoint')
            elif 'model' in weights and hasattr(weights['model'], 'state_dict'):
                source_state = weights['model'].state_dict()
                log('✅ Extracted state dict from nested model')
            else:
                source_state = weights
                log('✅ Using direct state dict')
            
            # Get current model state - MAKE A COPY TO MODIFY
            current_state = self.model.state_dict()
            modified_state = {}
            
            # Transfer compatible layers only
            transferred_layers = []
            skipped_layers = []
            total_transferred_params = 0
            
            # First, copy all current model parameters to modified_state
            for name, param in current_state.items():
                modified_state[name] = param.clone()
            
            # Then, selectively transfer compatible weights from source
            for name, param in source_state.items():
                # Skip embeddings due to different dataset sizes
                if name in ['uEmbeds', 'iEmbeds']:
                    skipped_layers.append(f"{name}: Different dataset size")
                    log(f"   ⚠️  Skipping {name}: Dataset size mismatch")
                    # Keep the current model's embeddings (already in modified_state)
                    continue
                
                if name in current_state:
                    if param.shape == current_state[name].shape:
                        # Compatible shape - transfer directly
                        modified_state[name] = param.clone().detach()
                        transferred_layers.append(name)
                        total_transferred_params += param.numel()
                        log(f"   ✅ Transferred: {name} {param.shape}")
                    else:
                        # Incompatible shape - skip
                        skipped_layers.append(f"{name}: {param.shape} -> {current_state[name].shape}")
                        log(f"   ⚠️  Skipping {name}: shape mismatch {param.shape} vs {current_state[name].shape}")
                else:
                    skipped_layers.append(f"{name}: not found in target model")
                    log(f"   ⚠️  Skipping {name}: not found in target model")
            
            # Load the modified state dict
            self.model.load_state_dict(modified_state)
            
            # Reinitialize embeddings for new dataset (but keep them trainable if not frozen)
            log("🔄 Reinitializing embeddings for new dataset...")
            with torch.no_grad():
                nn.init.xavier_uniform_(self.model.uEmbeds)
                nn.init.xavier_uniform_(self.model.iEmbeds)
            
            # Sync distillation model with the corrected model state
            self.distill_model.load_state_dict(modified_state, strict=False)

            log(f"✅ Transfer learning completed:")
            log(f"   📊 Transferred layers: {len(transferred_layers)}")
            log(f"   📊 Transferred parameters: {total_transferred_params:,}")
            log(f"   📊 Skipped layers: {len(skipped_layers)}")
            log(f"   🎯 Target dataset: {args.data}")
            log(f"   🎯 Embeddings reinitialized for dataset: {args.data}")
            
            # Show transferred layers summary
            if transferred_layers:
                log("   📋 Successfully transferred:")
                for layer in transferred_layers:
                    log(f"      • {layer}")
            
            return True
            
        except Exception as e:
            log(f'❌ Error in transfer learning: {e}')
            import traceback
            traceback.print_exc()
            return False

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        
        checkpoint_loaded = False

        if hasattr(args, 'load_weights') and args.load_weights:
            # First try regular loading
            checkpoint_loaded = self.load_model_weights(args.load_weights)
            
            # If regular loading fails due to size mismatch, try transfer learning
            if not checkpoint_loaded:
                log('🔄 Regular weight loading failed, attempting transfer learning...')
                checkpoint_loaded = self.load_model_weights_for_transfer(args.load_weights)
                
                if checkpoint_loaded:
                    log('✅ Transfer learning completed successfully')
                    
                    # Reapply freezing after transfer learning
                    if (hasattr(args, 'freeze_first_percent') and args.freeze_first_percent > 0) or \
                       (hasattr(args, 'freeze_last_percent') and args.freeze_last_percent > 0) or \
                       (hasattr(args, 'freeze_embeddings') and args.freeze_embeddings) or \
                       (hasattr(args, 'freeze_backbone') and args.freeze_backbone):
                        log("🔄 Reapplying freezing strategy after transfer learning...")
                        self.apply_freezing_strategy()
                        self.setup_fine_tuning_optimizer()
                        
            if checkpoint_loaded:
                # Set appropriate mode
                if args.epoch == 0:
                    self.model.eval()
                    self.distill_model.eval()
                    log('Models set to evaluation mode')
                else:
                    self.model.train()
                    self.distill_model.train()
                    log('Models set to training mode for fine-tuning')
            else:
                log('❌ Failed to load model weights (both regular and transfer learning)')

        elif hasattr(args, 'load_checkpoint') and args.load_checkpoint:
            # Load specific checkpoint file
            checkpoint_loaded = self.load_checkpoint(args.load_checkpoint)
            if checkpoint_loaded:
                log('Specific checkpoint loaded successfully')
                log(f'Loaded from: {args.load_checkpoint}')
            else:
                log('Failed to load specific checkpoint, starting fresh')
        
        elif hasattr(args, 'load_best') and args.load_best:
            # Load best checkpoint
            checkpoint_loaded = self.load_checkpoint(load_best=True)
            if checkpoint_loaded:
                log('Best checkpoint loaded successfully')
            else:
                log('No best checkpoint found, starting fresh')
        
        elif hasattr(args, 'resume') and args.resume:
            # Resume from latest checkpoint
            checkpoint_loaded = self.load_checkpoint()
            if checkpoint_loaded:
                log('Resumed from latest checkpoint')
            else:
                log('No checkpoint found, starting fresh')
        
        elif args.load_model != None:
            try:
                # Legacy model loading
                self.loadModel()
                checkpoint_loaded = True
                log('Legacy model loaded successfully')
            except Exception as e:
                log(f'Legacy model loading failed: {e}')
                checkpoint_loaded = False
        
        else:
            log('Model Initialized from scratch')
        
        bestRes = None
        result = []
        
        # Handle evaluation-only mode (epoch=0)
        if args.epoch == 0:
            log('Evaluation-only mode (epoch=0)')
            if not checkpoint_loaded:
                log('ERROR: No checkpoint loaded for evaluation!')
                log('Please provide a valid checkpoint path using --load_checkpoint')
                return
            
            # Set models to evaluation mode
            self.model.eval()
            self.distill_model.eval()
            
            # Run evaluation
            reses = self.testEpoch()
            log(self.makePrint('Evaluation', 0, reses, True))
            
            bestRes = reses
            
            # Save evaluation results
            torch.save([reses], f"Evaluation_result_{args.data}.pkl")
            log('Evaluation completed and results saved')
            
            if bestRes is not None:
                log(self.makePrint('Best Result', 0, bestRes, True))
            
            return
        
        # Training loop
        log(f"🚀 Starting training for {args.epoch} epochs...")
        
        for ep in range(self.start_epoch, args.epoch):
            # Set models to training mode
            self.model.train()
            self.distill_model.train()
            
            tstFlag = (ep % args.tstEpoch == 0)
            
            try:
                reses = self.trainEpoch()
                log(self.makePrint('Train', ep, reses, tstFlag))
            except Exception as e:
                log(f"❌ Training error at epoch {ep}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            if tstFlag:
                # Set models to evaluation mode for testing
                self.model.eval()
                self.distill_model.eval()
                
                try:
                    reses = self.valEpoch()
                    log(self.makePrint('Validation', ep, reses, tstFlag))
            
                    reses = self.testEpoch()
                    log(self.makePrint('Test', ep, reses, tstFlag))
                    
                    # Check if this is the best model
                    is_best = reses['Recall'] > self.best_recall
                    if is_best:
                        self.best_recall = reses['Recall']
                        self.best_ndcg = reses['NDCG']
                        bestRes = reses
                    
                    # Save checkpoint
                    if hasattr(args, 'save_freq') and (ep % args.save_freq == 0 or is_best):
                        self.save_checkpoint(ep, is_best=is_best)
                    else:
                        self.save_checkpoint(ep, is_best=is_best)
                    
                    # Save weights
                    if hasattr(args, 'save_weights_freq') and (ep % args.save_weights_freq == 0):
                        self.save_model_weights(ep)
                    elif ep % (args.tstEpoch * 2) == 0:
                        self.save_model_weights(ep)
                    
                    self.saveHistory()
                    result.append(reses)
                    
                    if bestRes is None:
                        bestRes = reses
                        
                except Exception as e:
                    log(f"❌ Testing error at epoch {ep}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save checkpoint every save_freq epochs
            elif hasattr(args, 'save_freq') and (ep % args.save_freq == 0):
                self.save_checkpoint(ep, is_best=False)
            elif ep % 10 == 0:
                self.save_checkpoint(ep, is_best=False)
            
            print()
        
        # Final evaluation and save
        if args.epoch > 0:
            try:
                self.model.eval()
                self.distill_model.eval()
                
                reses = self.testEpoch()
                result.append(reses)
                
                self.save_checkpoint(args.epoch - 1, is_final=True)
                torch.save(result, f"Transfer_result_{args.data}.pkl")
                
                log(self.makePrint('Final Test', args.epoch, reses, True))
                
                if bestRes is not None:
                    log(self.makePrint('Best Result', args.epoch, bestRes, True))
                else:
                    log('No best result available')
                
                self.saveHistory()
                
            except Exception as e:
                log(f"❌ Final evaluation error: {e}")
                import traceback
                traceback.print_exc()

    def trainEpoch(self):
        if hasattr(self, 'current_epoch') and hasattr(args, 'progressive_unfreeze') and args.progressive_unfreeze:
            self.progressive_unfreeze_layers(self.current_epoch, args.epoch)
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            # Générez les cibles de distillation
            with torch.no_grad():
                distill_usrEmbeds, distill_itmEmbeds, distill_cList, distill_subLst = self.distill_model(
                    self.handler, False, sub, cmp, encoderAdj, decoderAdj)

            usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            usrEmbeds2 = subLst[:args.user]
            itmEmbeds2 = subLst[args.user:]
            ancEmbeds2 = usrEmbeds2[ancs]
            posEmbeds2 = itmEmbeds2[poss]

            bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
            bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

            regLoss = calcRegLoss(self.model) * args.reg

            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(
                ancs,
                usrEmbeds,
                itmEmbeds) + args.ctra * contrastNCE(ancs, subLst, cList)

            # Calculez les pertes de distillation
            distill_loss_usr = F.mse_loss(usrEmbeds, distill_usrEmbeds)
            distill_loss_itm = F.mse_loss(itmEmbeds, distill_itmEmbeds)
            distill_loss_cList = F.mse_loss(cList, distill_cList)
            distill_loss_subLst = F.mse_loss(subLst, distill_subLst)

            # Combiner les pertes de distillation
            distill_loss = (distill_loss_usr + distill_loss_itm + distill_loss_cList + distill_loss_subLst) * self.distill_weight

            # Utilisez la perte de distillation pour la mise à jour du modèle
            loss = bprLoss + regLoss + contrastLoss + args.b2 * bprLoss2 + distill_loss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)

        # Mettez à jour le modèle de distillation
        self.distill_model.load_state_dict(self.model.state_dict())

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret
    
    def valEpoch(self):
        valLoader = self.handler.valLoader
        epLoss, epPreLoss = 0, 0
        steps = valLoader.dataset.__len__() // args.batch
        with torch.no_grad():
            for i, tem in enumerate(valLoader):
                if i % args.fixSteps == 0:
                    att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                    encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
                ancs, poss, negs = tem
                ancs = ancs.long().cuda()
                poss = poss.long().cuda()
                negs = negs.long().cuda()

                usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
                ancEmbeds = usrEmbeds[ancs]
                posEmbeds = itmEmbeds[poss]
                negEmbeds = itmEmbeds[negs]

                usrEmbeds2 = subLst[:args.user]
                itmEmbeds2 = subLst[args.user:]
                ancEmbeds2 = usrEmbeds2[ancs]
                posEmbeds2 = itmEmbeds2[poss]

                bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
                scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
                bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

                regLoss = calcRegLoss(self.model) * args.reg

                contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(
                    ancs, usrEmbeds, itmEmbeds) + args.ctra * contrastNCE(ancs, subLst, cList)
                loss = bprLoss + regLoss + contrastLoss + args.b2 * bprLoss2

                epLoss += loss.item()
                epPreLoss += bprLoss.item()
                log('Validation Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                    i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
                
        ret = dict()
        if steps > 0:
            ret['Loss'] = epLoss / steps
            ret['preLoss'] = epPreLoss / steps
        else:
            ret['Loss'] = 0
            ret['preLoss'] = 0
        return ret
            
    def testEpoch(self):
        # FIXED: Ensure model is in evaluation mode
        self.model.eval()
        self.distill_model.eval()
        
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        
        with torch.no_grad():
            for usr, trnMask in tstLoader:
                i += 1
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()
                usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj, self.handler.torchBiAdj,
                                                            self.handler.torchBiAdj)

                allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = torch.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                epRecall += recall
                epNdcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                    oneline=True)
        
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        torch.save(content, 'Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        # FIXED: Add weights_only=False for legacy model loading
        ckp = torch.load('Models/' + args.load_model + '.mod', weights_only=False)
        self.model = ckp['model']
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True

    log('Start')
    if torch.cuda.is_available():
        print("using cuda")
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()