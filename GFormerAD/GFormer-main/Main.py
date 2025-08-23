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
import torch as t

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
set_seed(getattr(args, 'seed', 42))

import os
import random
import numpy as np
import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
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
        self.frozen_layers = set()

    def get_ordered_parameters(self):
        ordered = []
        ordered.extend([('uEmbeds', self.model.uEmbeds), ('iEmbeds', self.model.iEmbeds)])
        for i, l in enumerate(self.model.gcnLayers):
            for n, p in l.named_parameters():
                ordered.append((f'gcnLayers.{i}.{n}', p))
        for n, p in self.model.gtLayers.named_parameters():
            ordered.append((f'gtLayers.{n}', p))
        for i, l in enumerate(self.model.pnnLayers):
            for n, p in l.named_parameters():
                ordered.append((f'pnnLayers.{i}.{n}', p))
        return ordered

    def freeze_first_percent(self, percent):
        if percent <= 0: return 0
        ordered = self.get_ordered_parameters()
        freeze_count = int(len(ordered) * percent)
        for i in range(freeze_count):
            name, param = ordered[i]
            param.requires_grad = False
            self.frozen_layers.add(name)
            log(f'Frozen: {name}')
        return freeze_count

    def freeze_last_percent(self, percent):
        if percent <= 0: return 0
        ordered = self.get_ordered_parameters()
        freeze_count = int(len(ordered) * percent)
        for i in range(len(ordered)-freeze_count, len(ordered)):
            name, param = ordered[i]
            param.requires_grad = False
            self.frozen_layers.add(name)
            log(f'Frozen: {name}')
        return freeze_count

    def freeze_backbone_keep_head(self):
        c = 0
        self.model.uEmbeds.requires_grad = False
        self.model.iEmbeds.requires_grad = False
        self.frozen_layers.update(['uEmbeds','iEmbeds'])
        c += 2
        for n,p in self.model.gcnLayers.named_parameters():
            p.requires_grad = False
            self.frozen_layers.add(f'gcnLayers.{n}')
            c += 1
        for n,p in self.model.gtLayers.named_parameters():
            p.requires_grad = False
            self.frozen_layers.add(f'gtLayers.{n}')
            c += 1
        return c

    def progressive_unfreeze_layers(self, current_epoch, total_epochs):
        if not args.progressive_unfreeze: return
        progress = current_epoch / total_epochs
        if args.unfreeze_schedule == 'exponential':
            target_ratio = progress ** 2
        else:
            target_ratio = progress
        ordered = self.get_ordered_parameters()
        frozen = [n for n in self.frozen_layers]
        if not frozen: return
        target_unfrozen = int(len(frozen) * target_ratio)
        unfrozen_now = 0
        for name, param in reversed(ordered):
            if name in self.frozen_layers and unfrozen_now < target_unfrozen:
                param.requires_grad = True
                self.frozen_layers.remove(name)
                unfrozen_now += 1
        if unfrozen_now > 0:
            self.update_optimizer_for_unfrozen_layers()
            log(f'Progressive unfreeze epoch {current_epoch}: {unfrozen_now} layers')

    def update_optimizer_for_unfrozen_layers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        base_lr = args.fine_tune_lr if getattr(args, 'fine_tune_lr', None) is not None else args.lr
        if hasattr(args, 'frozen_lr_scale'):
            base_lr *= args.frozen_lr_scale
        self.opt = torch.optim.Adam(trainable, lr=base_lr, weight_decay=0)

    def apply_freezing_strategy(self):
        if args.freeze_first_percent > 0:
            self.freeze_first_percent(args.freeze_first_percent)
        if args.freeze_last_percent > 0:
            self.freeze_last_percent(args.freeze_last_percent)
        if args.freeze_embeddings:
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.requires_grad = False
            self.frozen_layers.update(['uEmbeds','iEmbeds'])
        if args.freeze_backbone:
            self.freeze_backbone_keep_head()
        log(f'Frozen layers: {len(self.frozen_layers)}')

    def setup_fine_tuning_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        lr = args.fine_tune_lr if getattr(args,'fine_tune_lr',None) is not None else args.lr
        self.opt = torch.optim.Adam(trainable, lr=lr, weight_decay=0)

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.ResidualGTLayer).cuda()
        self.distill_model = Model(self.ResidualGTLayer).cuda()
        if (args.freeze_first_percent>0 or args.freeze_last_percent>0 or
            args.freeze_embeddings or args.freeze_backbone):
            self.apply_freezing_strategy()
        self.setup_fine_tuning_optimizer()
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
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        path = os.path.join(self.checkpoint_dir,
                            'final_checkpoint.pth' if is_final else f'checkpoint_epoch_{epoch}.pth')
        torch.save(ckp, path)
        if is_best:
            torch.save(ckp, os.path.join(self.best_checkpoint_dir, f'best_checkpoint_epoch_{epoch}.pth'))
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
                files = []
                if os.path.exists(self.best_checkpoint_dir):
                    files = [f for f in os.listdir(self.best_checkpoint_dir) if f.startswith('best_checkpoint_')]
                if not files: return False
                files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
                checkpoint_path = os.path.join(self.best_checkpoint_dir, files[0])
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir,'latest_checkpoint.pth')
        if not os.path.exists(checkpoint_path): return False
        try:
            ckp = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            self.model.load_state_dict(ckp['model_state_dict'])
            self.distill_model.load_state_dict(ckp['distill_model_state_dict'])
            self.gtLayer.load_state_dict(ckp['gtLayer_state_dict'])
            if args.epoch>0:
                self.opt.load_state_dict(ckp['optimizer_state_dict'])
            self.metrics = ckp['metrics']
            self.best_recall = ckp.get('best_recall',0.0)
            self.best_ndcg = ckp.get('best_ndcg',0.0)
            self.start_epoch = ckp['epoch'] + 1
            return True
        except Exception as e:
            log(f'Load checkpoint error: {e}')
            return False

    def save_model_weights(self, epoch, suffix=''):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')},
                   os.path.join('Models', f'weights_epoch_{epoch}{suffix}.pth'))

    def load_model_weights(self, path):
        if not os.path.exists(path): return False
        try:
            w = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            if 'model_state_dict' in w:
                self.model.load_state_dict(w['model_state_dict'])
            elif 'model' in w and hasattr(w['model'],'state_dict'):
                self.model.load_state_dict(w['model'].state_dict())
            else:
                self.model.load_state_dict(w)
            self.distill_model.load_state_dict(self.model.state_dict())
            self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
            return True
        except Exception as e:
            log(f'Load weights error: {e}')
            return False

    def load_model_weights_for_transfer(self, path):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            if 'model_state_dict' in ckpt:
                source = ckpt['model_state_dict']
            elif 'model' in ckpt and hasattr(ckpt['model'],'state_dict'):
                source = ckpt['model'].state_dict()
            else:
                source = ckpt
            cur = self.model.state_dict()
            new_state = {k:v.clone() for k,v in cur.items()}
            new_u = nn.Parameter(torch.empty(args.user, source['uEmbeds'].shape[1])); nn.init.xavier_uniform_(new_u)
            new_i = nn.Parameter(torch.empty(args.item, source['iEmbeds'].shape[1])); nn.init.xavier_uniform_(new_i)
            new_state['uEmbeds'] = new_u
            new_state['iEmbeds'] = new_i
            for k,v in source.items():
                if k in ['uEmbeds','iEmbeds']: continue
                if k in cur and v.shape == cur[k].shape:
                    new_state[k] = v.clone()
            self.model.load_state_dict(new_state, strict=False)
            self.distill_model.load_state_dict(self.model.state_dict())
            return True
        except Exception as e:
            log(f'Transfer load error: {e}')
            return False

    def debug_model_dimensions(self):
        log(f'uEmbeds: {getattr(self.model,"uEmbeds",None).shape if hasattr(self.model,"uEmbeds") else "NA"}')
        log(f'iEmbeds: {getattr(self.model,"iEmbeds",None).shape if hasattr(self.model,"iEmbeds") else "NA"}')
        if hasattr(self.handler,'torchBiAdj'):
            log(f'Adj shape: {self.handler.torchBiAdj.shape}')

    def makePrint(self, name, ep, res, save, total_epochs=None):
        total = total_epochs if total_epochs is not None else args.epoch
        out = f'Epoch {ep}/{total}, {name}: '
        for k,v in res.items():
            out += f'{k} = {v:.4f}, '
            key = name + k
            if save and key in self.metrics:
                self.metrics[key].append(v)
        return out[:-2]

    def run_with_curriculum(self):
        try:
            fracs = [float(f) for f in args.curriculum_schedule.split(',')]
            stages = [int(e) for e in args.curriculum_epochs.split(',')]
            if len(fracs)!=len(stages): return
        except:
            return
        total_epochs = sum(stages)
        g_ep = self.start_epoch
        bestRes = None
        results = []
        for si,f in enumerate(fracs):
            self.handler.trnLoader = self.handler.get_curriculum_loader(f)
            for _ in range(stages[si]):
                if g_ep >= total_epochs: break
                self.current_epoch = g_ep
                self.model.train(); self.distill_model.train()
                tr = self.trainEpoch()
                log(self.makePrint(f'Train(Stage{si+1})', g_ep, tr, True, total_epochs))
                if g_ep % args.tstEpoch == 0:
                    self.model.eval(); self.distill_model.eval()
                    vr = self.valEpoch()
                    log(self.makePrint('Validation', g_ep, vr, True, total_epochs))
                    te = self.testEpoch()
                    log(self.makePrint('Test', g_ep, te, True, total_epochs))
                    if te['Recall'] > self.best_recall:
                        self.best_recall = te['Recall']; self.best_ndcg = te['NDCG']; bestRes = te
                    self.save_checkpoint(g_ep, is_best=(te==bestRes))
                    if g_ep % (args.tstEpoch*2)==0:
                        self.save_model_weights(g_ep,'_curriculum')
                    self.saveHistory()
                    results.append(te)
                elif g_ep % 10 == 0:
                    self.save_checkpoint(g_ep, is_best=False)
                g_ep += 1
        self.model.eval(); self.distill_model.eval()
        final = self.testEpoch()
        results.append(final)
        self.save_checkpoint(total_epochs-1, is_final=True)
        torch.save(results, f'Curriculum_result_{args.data}.pkl')
        log(self.makePrint('Final Test', total_epochs, final, True, total_epochs))
        if bestRes:
            log(self.makePrint('Best Result', total_epochs, bestRes, True, total_epochs))
        self.saveHistory()

    def run_standard_training(self):
        bestRes = None
        results = []
        for ep in range(self.start_epoch, args.epoch):
            self.current_epoch = ep
            self.model.train(); self.distill_model.train()
            tstFlag = (ep % args.tstEpoch == 0)
            tr = self.trainEpoch()
            log(self.makePrint('Train', ep, tr, tstFlag))
            if tstFlag:
                self.model.eval(); self.distill_model.eval()
                vr = self.valEpoch(); log(self.makePrint('Validation', ep, vr, tstFlag))
                te = self.testEpoch(); log(self.makePrint('Test', ep, te, tstFlag))
                if te['Recall'] > self.best_recall:
                    self.best_recall = te['Recall']; self.best_ndcg = te['NDCG']; bestRes = te
                self.save_checkpoint(ep, is_best=(te==bestRes))
                if ep % (args.tstEpoch*2)==0:
                    self.save_model_weights(ep,'_standard')
                self.saveHistory()
                results.append(te)
                if bestRes is None: bestRes = te
                if getattr(args, 'early_stop', 0) > 0:
                    cur = te['Recall']
                    if not hasattr(self, '_es_best'):
                        self._es_best = cur; self._es_wait = 0
                    elif cur >= self._es_best + 0.0005:
                        self._es_best = cur; self._es_wait = 0
                    else:
                        self._es_wait += 1
                        if self._es_wait >= args.early_stop:
                            log(f'Early stopping triggered (best Recall={self._es_best:.4f}) at epoch {ep}')
                            break
            elif ep % 10 == 0:
                self.save_checkpoint(ep, is_best=False)
        if args.epoch>0:
            self.model.eval(); self.distill_model.eval()
            final = self.testEpoch()
            results.append(final)
            self.save_checkpoint(args.epoch-1, is_final=True)
            torch.save(results, f'Standard_result_{args.data}.pkl')
            log(self.makePrint('Final Test', args.epoch, final, True))
            if bestRes:
                log(self.makePrint('Best Result', args.epoch, bestRes, True))
            self.saveHistory()

    def run(self):
        self.prepareModel()
        checkpoint_loaded = False
        if getattr(args,'load_weights',None):
            checkpoint_loaded = self.load_model_weights(args.load_weights)
            if not checkpoint_loaded:
                checkpoint_loaded = self.load_model_weights_for_transfer(args.load_weights)
                if checkpoint_loaded and (args.freeze_first_percent>0 or args.freeze_last_percent>0 or
                                          args.freeze_embeddings or args.freeze_backbone):
                    self.apply_freezing_strategy(); self.setup_fine_tuning_optimizer()
            if checkpoint_loaded:
                if args.epoch==0:
                    self.model.eval(); self.distill_model.eval()
                else:
                    self.model.train(); self.distill_model.train()
        elif getattr(args,'load_checkpoint',None):
            checkpoint_loaded = self.load_checkpoint(args.load_checkpoint)
        elif getattr(args,'load_best',False):
            checkpoint_loaded = self.load_checkpoint(load_best=True)
        elif getattr(args,'resume',False):
            checkpoint_loaded = self.load_checkpoint()
        elif args.load_model is not None:
            try:
                self.loadModel(); checkpoint_loaded = True
            except:
                checkpoint_loaded = False
        if args.epoch==0:
            if not checkpoint_loaded:
                log('No checkpoint for evaluation'); return
            self.model.eval(); self.distill_model.eval()
            res = self.testEpoch()
            log(self.makePrint('Evaluation',0,res,True))
            torch.save([res], f'Evaluation_result_{args.data}.pkl')
            return
        self.debug_model_dimensions()
        if (self.model.uEmbeds.shape[0]!=args.user) or (self.model.iEmbeds.shape[0]!=args.item):
            new_u = nn.Parameter(torch.empty(args.user, self.model.uEmbeds.shape[1], device=self.model.uEmbeds.device))
            new_i = nn.Parameter(torch.empty(args.item, self.model.iEmbeds.shape[1], device=self.model.iEmbeds.device))
            nn.init.xavier_uniform_(new_u); nn.init.xavier_uniform_(new_i)
            self.model.uEmbeds = new_u; self.model.iEmbeds = new_i
            self.distill_model.uEmbeds = nn.Parameter(new_u.clone())
            self.distill_model.iEmbeds = nn.Parameter(new_i.clone())
            self.setup_fine_tuning_optimizer()
        if getattr(args,'curriculum',False):
            self.run_with_curriculum()
        else:
            self.run_standard_training()
        if getattr(args,'eval_emb_baselines',False):
            self.evaluate_embedding_baselines()

    def trainEpoch(self):
        if hasattr(self,'current_epoch') and getattr(args,'progressive_unfreeze',False):
            total_epochs = (sum([int(e) for e in args.curriculum_epochs.split(',')])
                            if getattr(args,'curriculum',False) else args.epoch)
            self.progressive_unfreeze_layers(self.current_epoch, total_epochs)
        total_epochs = (sum([int(e) for e in args.curriculum_epochs.split(',')])
                        if getattr(args,'curriculum',False) else args.epoch)
        temperature = self.calculate_temperature(getattr(self,'current_epoch',0), total_epochs)
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss = 0; epPre = 0
        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(), self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs = ancs.long().cuda(); poss = poss.long().cuda(); negs = negs.long().cuda()
            with torch.no_grad():
                d_u, d_i, d_c, d_s = self.distill_model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
            try:
                usrE, itmE, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj, temperature=temperature)
            except TypeError:
                usrE, itmE, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
            ancE = usrE[ancs]; posE = itmE[poss]; negE = itmE[negs]
            usrE2 = subLst[:args.user]; itmE2 = subLst[args.user:]
            ancE2 = usrE2[ancs]; posE2 = itmE2[poss]
            bpr1 = (-torch.sum(ancE * posE, dim=-1)).mean() * temperature
            scoreDiff = pairPredict(ancE2, posE2, negE)
            bpr2 = -(scoreDiff).sigmoid().log().sum() / args.batch * temperature
            regLoss = calcRegLoss(self.model) * args.reg
            clLoss = ((contrast(ancs, usrE) + contrast(poss, itmE)) * args.ssl_reg +
                      contrast(ancs, usrE, itmE) + args.ctra * contrastNCE(ancs, subLst, cList)) * temperature
            distill = (F.mse_loss(usrE,d_u)+F.mse_loss(itmE,d_i)+F.mse_loss(cList,d_c)+F.mse_loss(subLst,d_s))*self.distill_weight
            loss = bpr1 + regLoss + clLoss + args.b2 * bpr2 + distill
            epLoss += loss.item(); epPre += bpr1.item()
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),20)
            self.opt.step()
            log('Step %d/%d: loss=%.3f reg=%.3f cl=%.3f temp=%.3f  ' % (i, steps, loss, regLoss, clLoss, temperature),
                save=False, oneline=True)
        self.distill_model.load_state_dict(self.model.state_dict())
        return {'Loss': epLoss/steps, 'preLoss': epPre/steps, 'Temperature': temperature}

    def valEpoch(self):
        total_epochs = (sum([int(e) for e in args.curriculum_epochs.split(',')])
                        if getattr(args,'curriculum',False) else args.epoch)
        _ = self.calculate_temperature(getattr(self,'current_epoch',0), total_epochs)
        valLoader = self.handler.valLoader
        epLoss=0; epPre=0
        steps = valLoader.dataset.__len__() // args.batch
        with torch.no_grad():
            for i, tem in enumerate(valLoader):
                if i % args.fixSteps == 0:
                    att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(), self.handler)
                    encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
                ancs, poss, negs = tem
                ancs = ancs.long().cuda(); poss = poss.long().cuda(); negs = negs.long().cuda()
                usrE, itmE, cList, subLst = self.model(self.handler, False, sub, cmp, encoderAdj, decoderAdj)
                ancE = usrE[ancs]; posE = itmE[poss]; negE = itmE[negs]
                usrE2 = subLst[:args.user]; itmE2 = subLst[args.user:]
                ancE2 = usrE2[ancs]; posE2 = itmE2[poss]
                bpr1 = (-torch.sum(ancE * posE, dim=-1)).mean()
                scoreDiff = pairPredict(ancE2, posE2, negE)
                bpr2 = -(scoreDiff).sigmoid().log().sum()/args.batch
                regLoss = calcRegLoss(self.model) * args.reg
                clLoss = (contrast(ancs, usrE)+contrast(poss,itmE))*args.ssl_reg + contrast(ancs, usrE, itmE) + args.ctra*contrastNCE(ancs, subLst, cList)
                loss = bpr1 + regLoss + clLoss + args.b2*bpr2
                epLoss += loss.item(); epPre += bpr1.item()
                log('Val %d/%d: loss=%.3f reg=%.3f cl=%.3f  ' % (i, steps, loss, regLoss, clLoss),
                    save=False, oneline=True)
        return {'Loss': epLoss/steps if steps>0 else 0, 'preLoss': epPre/steps if steps>0 else 0}

    def calculate_temperature(self, epoch, total_epochs):
        if not getattr(args,'heating',False): return 1.0
        min_t = getattr(args,'min_temp',0.1)
        max_t = getattr(args,'max_temp',5.0)
        schedule = getattr(args,'temp_schedule','linear')
        progress = epoch / total_epochs
        if schedule=='linear':
            tval = min_t + (max_t-min_t)*progress
        elif schedule=='exponential':
            tval = min_t * (max_t/min_t) ** progress
        elif schedule=='step':
            if progress < 0.3: tval = min_t
            elif progress < 0.6: tval = (min_t+max_t)/2
            else: tval = max_t
        else:
            tval = min_t + (max_t-min_t)*progress
        log(f'Epoch {epoch}/{total_epochs} Temp={tval:.4f}', save=False)
        return tval

    def testEpoch(self):
        self.model.eval(); self.distill_model.eval()
        tstLoader = self.handler.tstLoader
        epRecall=0; epNdcg=0; i=0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        with torch.no_grad():
            for usr, trnMask in tstLoader:
                i+=1
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()
                usrE, itmE, _, _ = self.model(self.handler, True, self.handler.torchBiAdj,
                                              self.handler.torchBiAdj, self.handler.torchBiAdj)
                preds = torch.mm(usrE[usr], itmE.t()) * (1-trnMask) - trnMask*1e8
                _, topLocs = torch.topk(preds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                epRecall += recall; epNdcg += ndcg
                log('Test %d/%d: recall=%.2f ndcg=%.2f  ' % (i, steps, recall, ndcg), save=False, oneline=True)
        return {'Recall': epRecall/num, 'NDCG': epNdcg/num}

    def calcRes(self, topLocs, tstLocs, batIds):
        allR=0; allN=0
        for i in range(len(batIds)):
            tTop = list(topLocs[i])
            tTst = tstLocs[batIds[i]]
            tstNum = len(tTst)
            maxDcg = np.sum([np.reciprocal(np.log2(l+2)) for l in range(min(tstNum, args.topk))])
            rec=0; dcg=0
            for v in tTst:
                if v in tTop:
                    rec += 1
                    dcg += np.reciprocal(np.log2(tTop.index(v)+2))
            rec /= tstNum
            ndcg = dcg / maxDcg if maxDcg>0 else 0
            allR += rec; allN += ndcg
        return allR, allN

    def saveHistory(self):
        if args.epoch==0 and not getattr(args,'curriculum',False): return
        with open('History/'+args.save_path+'.his','wb') as fs:
            pickle.dump(self.metrics, fs)
        torch.save({'model': self.model}, 'Models/'+args.save_path+'.mod')
        log(f'Model Saved: {args.save_path}')

    def loadModel(self):
        ckp = torch.load('Models/'+args.load_model+'.mod', weights_only=False)
        self.model = ckp['model']
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        with open('History/'+args.load_model+'.his','rb') as fs:
            self.metrics = pickle.load(fs)

    # --- Embedding Baselines (MF / CF / NCF) ---

    def evaluate_with_embeddings(self, user_emb, item_emb, tag, scorer='dot', ncf_mlp=None):
        tstLoader = self.handler.tstLoader
        total_recall=0.0; total_ndcg=0.0
        num = tstLoader.dataset.__len__()
        user_emb = user_emb.cuda(); item_emb = item_emb.cuda()
        with torch.no_grad():
            for usr_batch, trnMask in tstLoader:
                usr_batch = usr_batch.cuda()
                trnMask = trnMask.cuda()
                uE = user_emb[usr_batch]
                if scorer == 'dot':
                    scores = torch.mm(uE, item_emb.t())
                else:  # NCF
                    prod = uE.unsqueeze(1) * item_emb.unsqueeze(0)  # (B,I,d)
                    scores = ncf_mlp(prod).squeeze(-1)              # (B,I)
                scores = scores * (1-trnMask) - trnMask*1e8
                _, topLocs = torch.topk(scores, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr_batch)
                total_recall += recall; total_ndcg += ndcg
        recall = total_recall / num
        ndcg = total_ndcg / num
        log(f'[Baseline {tag}] Recall={recall:.4f}, NDCG={ndcg:.4f}')
        return {'Recall': recall, 'NDCG': ndcg}

    def train_mf_baseline(self):
        d = args.mf_dim if args.mf_dim else args.latdim
        U = torch.randn(args.user, d, device='cuda') * 0.01
        V = torch.randn(args.item, d, device='cuda') * 0.01
        U.requires_grad_(True); V.requires_grad_(True)
        opt = torch.optim.Adam([U,V], lr=args.mf_lr)
        rows = self.handler.trnLoader.dataset.rows
        cols = self.handler.trnLoader.dataset.cols
        n = len(rows); idx = np.arange(n)
        for ep in range(args.mf_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, args.baseline_batch):
                bidx = idx[s:s+args.baseline_batch]
                u = torch.as_tensor(rows[bidx], device='cuda', dtype=torch.long)
                i = torch.as_tensor(cols[bidx], device='cuda', dtype=torch.long)
                negs=[]
                for uu in u.tolist():
                    for _ in range(args.mf_neg):
                        while True:
                            ni = np.random.randint(args.item)
                            if ni not in self.handler.user_items[uu]:
                                negs.append(ni); break
                neg = torch.as_tensor(negs, device='cuda', dtype=torch.long)
                u_rep = u.repeat_interleave(args.mf_neg)
                pos = (U[u]*V[i]).sum(-1)
                negs = (U[u_rep]*V[neg]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos.repeat_interleave(args.mf_neg)-negs)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            log(f'MF epoch {ep+1}/{args.mf_epochs}')
        return U.detach(), V.detach()

    def build_cf_embeddings(self):
        d = args.latdim
        item_emb = torch.randn(args.item, d, device='cuda') * 0.01
        user_emb = torch.zeros(args.user, d, device='cuda')
        for u, items in enumerate(self.handler.user_items):
            if items:
                user_emb[u] = item_emb[items].mean(0)
        return user_emb, item_emb

    def train_ncf(self, U_init, V_init):
        U = U_init.clone().detach().requires_grad_(True)
        V = V_init.clone().detach().requires_grad_(True)
        hidden = [int(h) for h in args.ncf_hidden.split(',') if h]
        layers=[]; in_d = U.shape[1]
        for h in hidden:
            layers.append(nn.Linear(in_d,h)); layers.append(nn.ReLU()); in_d = h
        layers.append(nn.Linear(in_d,1))
        mlp = nn.Sequential(*layers).cuda()
        opt = torch.optim.Adam(list(mlp.parameters())+[U,V], lr=args.ncf_lr)
        rows = self.handler.trnLoader.dataset.rows
        cols = self.handler.trnLoader.dataset.cols
        n = len(rows); idx = np.arange(n)
        for ep in range(args.ncf_epochs):
            np.random.shuffle(idx)
            for s in range(0,n,args.baseline_batch):
                bidx = idx[s:s+args.baseline_batch]
                u = torch.as_tensor(rows[bidx], device='cuda', dtype=torch.long)
                i = torch.as_tensor(cols[bidx], device='cuda', dtype=torch.long)
                negs=[]
                for uu in u.tolist():
                    for _ in range(args.mf_neg):
                        while True:
                            ni = np.random.randint(args.item)
                            if ni not in self.handler.user_items[uu]:
                                negs.append(ni); break
                neg = torch.as_tensor(negs, device='cuda', dtype=torch.long)
                u_rep = u.repeat_interleave(args.mf_neg)
                pos_feat = (U[u]*V[i])
                neg_feat = (U[u_rep]*V[neg])
                pos_score = mlp(pos_feat).view(-1)
                neg_score = mlp(neg_feat).view(-1)
                loss = -torch.log(torch.sigmoid(pos_score.repeat_interleave(args.mf_neg)-neg_score)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            log(f'NCF epoch {ep+1}/{args.ncf_epochs}')
        return U.detach(), V.detach(), mlp

    def evaluate_embedding_baselines(self):
        selected = set([s.strip().upper() for s in getattr(args,'baseline_list','MF,CF,NCF').split(',') if s.strip()])
        log('=== Embedding Baselines Selected: ' + ', '.join(sorted(selected)) + ' ===')
        U_mf = V_mf = None
        if 'MF' in selected:
            U_mf, V_mf = self.train_mf_baseline()
            self.evaluate_with_embeddings(U_mf, V_mf, 'MF')
        if 'CF' in selected:
            U_cf, V_cf = self.build_cf_embeddings()
            self.evaluate_with_embeddings(U_cf, V_cf, 'CF')
        if 'NCF' in selected:
            if U_mf is None:  # need MF init
                U_mf, V_mf = self.train_mf_baseline()
            U_ncf, V_ncf, mlp = self.train_ncf(U_mf, V_mf)
            class ApplyLayers(nn.Module):
                def __init__(self, seq):
                    super().__init__(); self.seq = seq
                def forward(self,x):
                    B,I,D = x.shape
                    x = x.view(B*I, D)
                    x = self.seq(x)
                    return x.view(B,I,1)
            ncf_mlp = ApplyLayers(mlp).cuda()
            self.evaluate_with_embeddings(U_ncf, V_ncf, 'NCF', scorer='ncf', ncf_mlp=ncf_mlp)
        log('=== Baselines Done ===')


if __name__ == '__main__':
    logger.saveDefault = True
    log('Start')
    if torch.cuda.is_available():
        print('using cuda')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()