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
from datetime import datetime
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.distill_weight = 0.1
        self.ResidualGTLayer = ResidualGTLayer()
        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = {}
        for met in ['Loss', 'preLoss', 'Recall', 'NDCG']:
            self.metrics['Train' + met] = []
            self.metrics['Test' + met] = []
        self.checkpoint_dir = 'Checkpoints'
        self.best_checkpoint_dir = 'BestCheckpoints'
        self.create_checkpoint_dirs()
        self.best_recall = 0.0
        self.best_ndcg = 0.0
        self.start_epoch = 0

    def setup_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.ResidualGTLayer).cuda()
        self.distill_model = Model(self.ResidualGTLayer).cuda()
        self.setup_optimizer()
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def create_checkpoint_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_checkpoint_dir, exist_ok=True)
        os.makedirs('Models', exist_ok=True)
        os.makedirs('History', exist_ok=True)

    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        ckp = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'distill_model_state_dict': self.distill_model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'gtLayer_state_dict': self.gtLayer.state_dict(),
            'metrics': self.metrics,
            'best_recall': self.best_recall,
            'best_ndcg': self.best_ndcg,
            'args': vars(args),
        }
        path = os.path.join(self.checkpoint_dir,
                            'final_checkpoint.pth' if is_final else f'checkpoint_epoch_{epoch}.pth')
        torch.save(ckp, path)
        if is_best:
            torch.save(ckp, os.path.join(self.best_checkpoint_dir, 'best_checkpoint.pth'))
        torch.save(ckp, os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth'))
        self.cleanup_old_checkpoints(keep_last=getattr(args,'keep_checkpoints',5))

    def cleanup_old_checkpoints(self, keep_last=5):
        try:
            files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            for f in files[:-keep_last]:
                os.remove(os.path.join(self.checkpoint_dir,f))
        except Exception as e:
            log(f'Cleanup error: {e}')

    def load_checkpoint(self, checkpoint_path=None, load_best=False):
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = os.path.join(self.best_checkpoint_dir, 'best_checkpoint.pth')
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir,'latest_checkpoint.pth')
        if not os.path.exists(checkpoint_path): return False
        try:
            ckp = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            self.model.load_state_dict(ckp['model_state_dict'])
            self.distill_model.load_state_dict(ckp['distill_model_state_dict'])
            self.gtLayer.load_state_dict(ckp['gtLayer_state_dict'])
            if args.epoch > 0:
                self.opt.load_state_dict(ckp['optimizer_state_dict'])
            self.metrics = ckp['metrics']
            self.best_recall = ckp.get('best_recall',0.0)
            self.best_ndcg = ckp.get('best_ndcg',0.0)
            self.start_epoch = ckp['epoch'] + 1
            log(f"Resuming from epoch {self.start_epoch} from {checkpoint_path}")
            return True
        except Exception as e:
            log(f'Load checkpoint error: {e}')
            return False

    def load_model_weights(self, path):
        if not os.path.exists(path): return False
        try:
            w = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            state_dict = w['model_state_dict'] if 'model_state_dict' in w else w
            self.model.load_state_dict(state_dict)
            self.distill_model.load_state_dict(self.model.state_dict())
            self.setup_optimizer()
            log(f"Loaded model weights from {path}")
            return True
        except Exception as e:
            log(f'Load weights error: {e}')
            return False

    def makePrint(self, name, ep, res, save):
        out = f'Epoch {ep}/{args.epoch}, {name}: '
        for k,v in res.items():
            out += f'{k} = {v:.4f}, '
            key = name + k
            if save and key in self.metrics:
                self.metrics[key].append(v)
        return out[:-2]

    def run(self):
        self.prepareModel()
        
        if getattr(args, 'load_weights', None):
            self.load_model_weights(args.load_weights)
        elif getattr(args, 'resume', False):
            self.load_checkpoint()

        if args.epoch == 0:
            log("Epoch is 0, running evaluation only.")
            self.model.eval()
            res = self.testEpoch()
            log(self.makePrint('Evaluation', 0, res, True))
            return

        bestRes = None
        for ep in range(self.start_epoch, args.epoch):
            self.current_epoch = ep
            
            self.model.train()
            self.distill_model.load_state_dict(self.model.state_dict())
            
            tstFlag = (ep % args.tstEpoch == 0)
            tr = self.trainEpoch()
            log(self.makePrint('Train', ep, tr, tstFlag))

            if tstFlag:
                self.model.eval()
                vr = self.valEpoch()
                log(self.makePrint('Validation', ep, vr, tstFlag))
                te = self.testEpoch()
                log(self.makePrint('Test', ep, te, tstFlag))
                
                is_best = te['Recall'] > self.best_recall
                if is_best:
                    self.best_recall = te['Recall']
                    self.best_ndcg = te['NDCG']
                    bestRes = te
                
                self.save_checkpoint(ep, is_best=is_best)
                self.saveHistory()

                if bestRes is None: bestRes = te

                if getattr(args, 'early_stop', 0) > 0:
                    if not hasattr(self, '_es_best'): self._es_best = te['Recall']; self._es_wait = 0
                    elif te['Recall'] >= self._es_best + 0.0005: self._es_best = te['Recall']; self._es_wait = 0
                    else:
                        self._es_wait += 1
                        if self._es_wait >= args.early_stop:
                            log(f'Early stopping at epoch {ep}')
                            break
            elif ep % 10 == 0:
                self.save_checkpoint(ep, is_best=False)

        if args.epoch > 0:
            self.model.eval()
            final = self.testEpoch()
            self.save_checkpoint(args.epoch - 1, is_final=True)
            log(self.makePrint('Final Test', args.epoch, final, True))
            if bestRes: log(self.makePrint('Best Result', args.epoch, bestRes, True))
            self.saveHistory()

    def trainEpoch(self):
        temperature = self.calculate_temperature(self.current_epoch, args.epoch)
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(), self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs, poss, negs = ancs.long().cuda(), poss.long().cuda(), negs.long().cuda()
            
            with torch.no_grad():
                d_u, d_i, d_c, d_s = self.distill_model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
            
            usrE, itmE, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
            
            ancE, posE, negE = usrE[ancs], itmE[poss], itmE[negs]
            usrE2, itmE2 = subLst[:args.user], subLst[args.user:]
            ancE2, posE2 = usrE2[ancs], itmE2[poss]

            # --- 1. Calculate all raw loss components first ---
            bpr1_raw = (-torch.sum(ancE * posE, dim=-1)).mean()
            scoreDiff = pairPredict(ancE2, posE2, negE)
            bpr2_raw = -(scoreDiff).sigmoid().log().sum() / args.batch
            regLoss_raw = calcRegLoss(self.model) * args.reg
            clLoss_raw = (contrast(ancs, usrE) + contrast(poss, itmE)) * args.ssl_reg + \
                         contrast(ancs, usrE, itmE) + args.ctra * contrastNCE(ancs, subLst, cList)
            distill_raw = (F.mse_loss(usrE,d_u) + F.mse_loss(itmE,d_i) + \
                           F.mse_loss(cList,d_c) + F.mse_loss(subLst,d_s))

            # --- 2. Apply temperature scaling and weights uniformly (Original Logic) ---
            T = temperature
            bpr1 = bpr1_raw 
            bpr2 = bpr2_raw * args.b2 * T
            clLoss = clLoss_raw * T
            regLoss = regLoss_raw 
            distill = distill_raw * self.distill_weight * T

            # --- 3. Sum up the final loss ---
            loss = bpr1 + regLoss + clLoss + bpr2 + distill

            epLoss += loss.item()
            epPreLoss += bpr1.item()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 20)
            self.opt.step()
            
            if i % 50 == 0:
                log(f'Step {i}/{steps}: loss={loss:.3f} temp={temperature:.3f}', save=False, oneline=True)

        return {'Loss': epLoss/steps, 'preLoss': epPreLoss/steps, 'Temperature': temperature}

    def valEpoch(self):
        valLoader = self.handler.valLoader
        epLoss, epPreLoss = 0, 0
        steps = valLoader.dataset.__len__() // args.batch
        with torch.no_grad():
            for i, tem in enumerate(valLoader):
                if i % args.fixSteps == 0:
                    att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(), self.handler)
                    encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
                ancs, poss, negs = tem
                ancs, poss, negs = ancs.long().cuda(), poss.long().cuda(), negs.long().cuda()
                usrE, itmE, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
                ancE, posE, negE = usrE[ancs], itmE[poss], itmE[negs]
                usrE2, itmE2 = subLst[:args.user], subLst[args.user:]
                ancE2, posE2 = usrE2[ancs], itmE2[poss]
                
                bpr1 = (-torch.sum(ancE * posE, dim=-1)).mean()
                scoreDiff = pairPredict(ancE2, posE2, negE)
                bpr2 = -(scoreDiff).sigmoid().log().sum()/args.batch
                regLoss = calcRegLoss(self.model) * args.reg
                clLoss = (contrast(ancs, usrE)+contrast(poss,itmE))*args.ssl_reg + contrast(ancs, usrE, itmE) + args.ctra*contrastNCE(ancs, subLst, cList)
                
                loss = bpr1 + args.b2*bpr2 + regLoss + clLoss
                epLoss += loss.item()
                epPreLoss += bpr1.item()
        return {'Loss': epLoss/steps if steps>0 else 0, 'preLoss': epPreLoss/steps if steps>0 else 0}

    def calculate_temperature(self, epoch, total_epochs):
        if not getattr(args,'heating',False): return 1.0
        min_t, max_t = getattr(args,'min_temp',0.1), getattr(args,'max_temp',5.0)
        progress = epoch / total_epochs if total_epochs > 0 else 0
        # Exponential schedule
        tval = min_t * (max_t/min_t) ** progress if min_t > 0 else max_t * progress
        return tval

    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, i = 0, 0, 0
        num = tstLoader.dataset.__len__()
        with torch.no_grad():
            for usr, trnMask in tstLoader:
                i += 1
                usr, trnMask = usr.long().cuda(), trnMask.cuda()
                usrE, itmE, _, _ = self.model(self.handler, True, self.handler.torchBiAdj,
                                              self.handler.torchBiAdj, self.handler.torchBiAdj)
                preds = torch.mm(usrE[usr], itmE.t()) * (1-trnMask) - trnMask*1e8
                _, topLocs = torch.topk(preds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                epRecall += recall
                epNdcg += ndcg
        return {'Recall': epRecall/num, 'NDCG': epNdcg/num}

    def calcRes(self, topLocs, tstLocs, batIds):
        allR, allN = 0, 0
        for i in range(len(batIds)):
            tTop = list(topLocs[i])
            tTst = tstLocs[batIds[i]]
            tstNum = len(tTst)
            if tstNum == 0: continue
            maxDcg = np.sum([np.reciprocal(np.log2(l+2)) for l in range(min(tstNum, args.topk))])
            rec, dcg = 0, 0
            for v in tTst:
                if v in tTop:
                    rec += 1
                    dcg += np.reciprocal(np.log2(tTop.index(v)+2))
            allR += rec / tstNum
            allN += dcg / maxDcg if maxDcg > 0 else 0
        return allR, allN

    def saveHistory(self):
        if args.epoch==0: return
        with open('History/'+args.save_path+'.his','wb') as fs:
            pickle.dump(self.metrics, fs)
        torch.save({'model_state_dict': self.model.state_dict()}, 'Models/'+args.save_path+'.mod')
        log(f'Model Saved: {args.save_path}')

if __name__ == '__main__':
    logger.saveDefault = True
    log('Start')
    if torch.cuda.is_available(): print('using cuda')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()