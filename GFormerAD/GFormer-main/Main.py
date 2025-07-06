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
            checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
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
        """Load only model weights"""
        if not os.path.exists(weights_path):
            log(f'Weights file not found: {weights_path}')
            return False
        
        try:
            weights = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(weights['model_state_dict'])
            log(f'Model weights loaded: {weights_path}')
            return True
        except Exception as e:
            log(f'Error loading weights: {e}')
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
        
        # FIXED: Updated checkpoint loading logic to handle new parameters
        checkpoint_loaded = False
        
        if hasattr(args, 'load_checkpoint') and args.load_checkpoint:
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
            # Legacy model loading
            self.loadModel()
            checkpoint_loaded = True
            log('Legacy model loaded successfully')
        
        else:
            log('Model Initialized')
        
        bestRes = None
        result = []
        
        # FIXED: Handle evaluation-only mode (epoch=0) with better error handling
        if args.epoch == 0:
            log('Evaluation-only mode (epoch=0)')
            if not checkpoint_loaded:
                log('ERROR: No checkpoint loaded for evaluation!')
                log('Please provide a valid checkpoint path using --load_checkpoint')
                log('Example: --load_checkpoint /path/to/your/checkpoint.pth')
                return
            
            # Set models to evaluation mode
            self.model.eval()
            self.distill_model.eval()
            
            # Run evaluation
            reses = self.testEpoch()
            log(self.makePrint('Evaluation', 0, reses, True))
            
            # FIXED: Set bestRes for evaluation-only mode
            bestRes = reses
            
            # Save evaluation results
            torch.save([reses], f"Evaluation_result_{args.data}.pkl")
            log('Evaluation completed and results saved')
            
            # FIXED: Print best results for evaluation-only mode
            if bestRes is not None:
                log(self.makePrint('Best Result', 0, bestRes, True))
            
            return
        
        # Rest of the training loop remains the same...
        for ep in range(self.start_epoch, args.epoch):
            # Set models to training mode
            self.model.train()
            self.distill_model.train()
            
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            
            if tstFlag:
                # Set models to evaluation mode for validation/testing
                self.model.eval()
                self.distill_model.eval()
                
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
                
                # Save checkpoint based on save_freq parameter
                if hasattr(args, 'save_freq') and (ep % args.save_freq == 0 or is_best):
                    self.save_checkpoint(ep, is_best=is_best)
                else:
                    # Default behavior - save every test epoch
                    self.save_checkpoint(ep, is_best=is_best)
                
                # Save model weights based on save_weights_freq parameter
                if hasattr(args, 'save_weights_freq') and (ep % args.save_weights_freq == 0):
                    self.save_model_weights(ep)
                elif ep % (args.tstEpoch * 2) == 0:
                    # Default behavior
                    self.save_model_weights(ep)
                
                self.saveHistory()
                result.append(reses)
                
                if bestRes is None:
                    bestRes = reses
            
            # Save checkpoint every save_freq epochs (not just test epochs)
            elif hasattr(args, 'save_freq') and (ep % args.save_freq == 0):
                self.save_checkpoint(ep, is_best=False)
            elif ep % 10 == 0:
                # Default behavior
                self.save_checkpoint(ep, is_best=False)
            
            print()
        
        # Final evaluation and save (only for training mode)
        if args.epoch > 0:
            # Set models to evaluation mode for final test
            self.model.eval()
            self.distill_model.eval()
            
            reses = self.testEpoch()
            result.append(reses)
            
            # Save final checkpoint and results
            self.save_checkpoint(args.epoch - 1, is_final=True)
            torch.save(result, "Saeg_result.pkl")
            
            log(self.makePrint('Test', args.epoch, reses, True))
            
            # FIXED: Only print best results if bestRes exists
            if bestRes is not None:
                log(self.makePrint('Best Result', args.epoch, bestRes, True))
            else:
                log('No best result available (training was too short)')
            
            self.saveHistory()

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.ResidualGTLayer).cuda()
        self.distill_model = Model(self.ResidualGTLayer).cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def trainEpoch(self):
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
        ckp = torch.load('Models/' + args.load_model + '.mod')
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