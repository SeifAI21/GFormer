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
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.distill_weight = 0.1
        self.ResidualGTLayer=ResidualGTLayer()
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
        """Load checkpoint from file"""
        if checkpoint_path is None:
            if load_best:
                # Find the best checkpoint
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
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.gtLayer.load_state_dict(checkpoint['gtLayer_state_dict'])
            
            # Load training progress
            self.metrics = checkpoint['metrics']
            self.best_recall = checkpoint.get('best_recall', 0.0)
            self.best_ndcg = checkpoint.get('best_ndcg', 0.0)
            self.start_epoch = checkpoint['epoch'] + 1
            
            log(f'Checkpoint loaded: {checkpoint_path}')
            log(f'Resuming from epoch {self.start_epoch}')
            log(f'Best Recall: {self.best_recall:.4f}, Best NDCG: {self.best_ndcg:.4f}')
            
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
        
        # Load checkpoint if specified
        if args.load_model != None:
            if self.load_checkpoint(args.load_model):
                log('Checkpoint loaded successfully')
            else:
                log('Failed to load checkpoint, starting fresh')
        elif hasattr(args, 'resume') and args.resume:
            if self.load_checkpoint():
                log('Resumed from latest checkpoint')
            else:
                log('No checkpoint found, starting fresh')
        else:
            log('Model Initialized')
        
        bestRes = None
        result = []
        
        for ep in range(self.start_epoch, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            
            if tstFlag:
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
                self.save_checkpoint(ep, is_best=is_best)
                
                # Save model weights every few epochs
                if ep % (args.tstEpoch * 2) == 0:
                    self.save_model_weights(ep)
                
                self.saveHistory()
                result.append(reses)
                
                if bestRes is None:
                    bestRes = reses
            
            # Save checkpoint every few epochs (not just test epochs)
            if ep % 10 == 0:
                self.save_checkpoint(ep, is_best=False)
            
            print()
        
        # Final evaluation and save
        reses = self.testEpoch()
        result.append(reses)
        
        # Save final checkpoint and results
        self.save_checkpoint(args.epoch - 1, is_final=True)
        torch.save(result, "Saeg_result.pkl")
        
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('Best Result', args.epoch, bestRes, True))
        self.saveHistory()

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.ResidualGTLayer).cuda()
        self.distill_model = Model(self.ResidualGTLayer).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    # ... rest of your existing methods remain the same ...
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
            #loss = regLoss + contrastLoss + distill_loss
            epLoss += loss.item()
            #epPreLoss += bprLoss.item()
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
        valLoader = self.handler.valLoader  # Assurez-vous d'avoir un DataLoader pour les données de validation
        epLoss, epPreLoss = 0, 0
       # epRecall, epNdcg = 0, 0 
        steps = valLoader.dataset.__len__() // args.batch
        with torch.no_grad():  # Pas besoin de calculer les gradients pendant la validation
            for i, tem in enumerate(valLoader):
                if i % args.fixSteps == 0:
                    att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                    encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
                ancs, poss, negs = tem
                ancs = ancs.long().cuda()
                poss = poss.long().cuda()
                negs = negs.long().cuda()

                usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp,  encoderAdj,
                                                                           decoderAdj)
                ancEmbeds = usrEmbeds[ancs]
                posEmbeds = itmEmbeds[poss]
                negEmbeds = itmEmbeds[negs]

                usrEmbeds2 = subLst[:args.user]
                itmEmbeds2 = subLst[args.user:]
                ancEmbeds2 = usrEmbeds2[ancs]
                posEmbeds2 = itmEmbeds2[poss]

                bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
                #
                scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
                bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

                regLoss = calcRegLoss(self.model) * args.reg

                contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(
                ancs,
                usrEmbeds,
                itmEmbeds) + args.ctra*contrastNCE(ancs, subLst, cList)
                loss = bprLoss + regLoss + contrastLoss + args.b2*bprLoss2

                epLoss += loss.item()
                epPreLoss += bprLoss.item()
                log('Validation Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
            ret = dict()
            if steps > 0:
                ret['Loss'] = epLoss / steps
            else:
                ret['Loss'] = 0  # ou une autre valeur par défaut
            if steps > 0:
                ret['preLoss'] = epPreLoss / steps
            else:
                ret['preLoss'] = 0  # ou une autre valeur par défaut

            return ret
            
    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj, self.handler.torchBiAdj,
                                                          self.handler.torchBiAdj)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
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
        t.save(content, 'Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True

    log('Start')
    if t.cuda.is_available():
        print("using cuda")
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()