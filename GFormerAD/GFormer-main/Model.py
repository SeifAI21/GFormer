import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

# Fixed UltraGCN Layer Implementation
class UltraGCNLayer(nn.Module):
    def __init__(self):
        super(UltraGCNLayer, self).__init__()
        # UltraGCN uses direct optimization without message passing
        self.beta = nn.Parameter(torch.tensor(args.ultra_beta))
        self.gamma = nn.Parameter(torch.tensor(args.ultra_gamma))
        
    def compute_degrees_from_sparse(self, adj):
        """Compute user and item degrees from sparse adjacency matrix"""
        # Get edges
        rows, cols = adj._indices()[0, :], adj._indices()[1, :]
        
        # Count degrees by user/item type
        user_mask = rows < args.user
        item_mask = rows >= args.user
        
        # User interactions (users -> items)
        user_interactions = user_mask & (cols >= args.user)
        if user_interactions.sum() > 0:
            user_rows = rows[user_interactions]
            user_degrees = torch.zeros(args.user, device=adj.device)
            user_degrees.scatter_add_(0, user_rows, torch.ones_like(user_rows, dtype=torch.float))
        else:
            user_degrees = torch.ones(args.user, device=adj.device)
        
        # Item interactions (items -> users) 
        item_interactions = item_mask & (cols < args.user)
        if item_interactions.sum() > 0:
            item_rows = rows[item_interactions] - args.user
            item_degrees = torch.zeros(args.item, device=adj.device)
            item_degrees.scatter_add_(0, item_rows, torch.ones_like(item_rows, dtype=torch.float))
        else:
            item_degrees = torch.ones(args.item, device=adj.device)
        
        # Avoid division by zero
        user_degrees = torch.clamp(user_degrees, min=1e-8)
        item_degrees = torch.clamp(item_degrees, min=1e-8)
        
        return user_degrees, item_degrees
        
    def forward(self, adj, embeds):
        """
        Fixed UltraGCN forward pass using proper sparse operations
        """
        user_embeds = embeds[:args.user]
        item_embeds = embeds[args.user:]
        
        # Compute degrees
        user_degrees, item_degrees = self.compute_degrees_from_sparse(adj)
        
        # Get interaction edges
        rows, cols = adj._indices()[0, :], adj._indices()[1, :]
        
        # User -> Item interactions
        user_item_mask = (rows < args.user) & (cols >= args.user)
        if user_item_mask.sum() > 0:
            ui_users = rows[user_item_mask]
            ui_items = cols[user_item_mask] - args.user
            
            # Aggregate item embeddings to users
            user_agg = torch.zeros_like(user_embeds)
            user_agg.scatter_add_(0, ui_users.unsqueeze(1).expand(-1, args.latdim), 
                                 item_embeds[ui_items] / user_degrees[ui_users].unsqueeze(1))
        else:
            user_agg = torch.zeros_like(user_embeds)
        
        # Item -> User interactions  
        item_user_mask = (rows >= args.user) & (cols < args.user)
        if item_user_mask.sum() > 0:
            iu_items = rows[item_user_mask] - args.user
            iu_users = cols[item_user_mask]
            
            # Aggregate user embeddings to items
            item_agg = torch.zeros_like(item_embeds)
            item_agg.scatter_add_(0, iu_items.unsqueeze(1).expand(-1, args.latdim),
                                 user_embeds[iu_users] / item_degrees[iu_items].unsqueeze(1))
        else:
            item_agg = torch.zeros_like(item_embeds)
        
        # Combine with original embeddings using learnable beta
        enhanced_user_embeds = (1 - self.beta) * user_embeds + self.beta * user_agg
        enhanced_item_embeds = (1 - self.beta) * item_embeds + self.beta * item_agg
        
        return torch.cat([enhanced_user_embeds, enhanced_item_embeds], dim=0)

# Alternative: Simpler and More Stable UltraGCN
class SimpleUltraGCNLayer(nn.Module):
    def __init__(self):
        super(SimpleUltraGCNLayer, self).__init__()
        # Simplified version that's guaranteed to work
        self.alpha = nn.Parameter(torch.tensor(args.ultra_beta))
        
    def forward(self, adj, embeds):
        """Simple UltraGCN: weighted self + neighbor aggregation"""
        try:
            # Standard sparse matrix multiplication for neighbor aggregation
            neighbor_agg = torch.spmm(adj, embeds)
            
            # UltraGCN style: learnable combination
            output = self.alpha * embeds + (1 - self.alpha) * neighbor_agg
            
            return output
        except Exception as e:
            print(f"UltraGCN error: {e}, falling back to GCN")
            return torch.spmm(adj, embeds)

# Fallback: Enhanced GCN Layer (if UltraGCN fails)
class EnhancedGCNLayer(nn.Module):
    def __init__(self):
        super(EnhancedGCNLayer, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, adj, embeds):
        """Enhanced GCN with learnable weights and dropout"""
        neighbor_agg = torch.spmm(adj, embeds)
        output = self.weight * neighbor_agg
        return self.dropout(output)

# Updated Model class with error handling
class Model(nn.Module):
    def __init__(self, gtLayer):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        
        # Choose layer type with fallback options
        if getattr(args, 'use_ultragcn', False):
            print("🔥 Using UltraGCN layers")
            try:
                if getattr(args, 'simple_ultra', False):
                    print("  -> Simple UltraGCN variant")
                    self.gcnLayers = nn.ModuleList([
                        SimpleUltraGCNLayer() for _ in range(args.gcn_layer)
                    ])
                else:
                    print("  -> Full UltraGCN implementation")
                    self.gcnLayers = nn.ModuleList([
                        UltraGCNLayer() for _ in range(args.gcn_layer)
                    ])
            except Exception as e:
                print(f"UltraGCN initialization failed: {e}")
                print("Falling back to Enhanced GCN")
                self.gcnLayers = nn.ModuleList([
                    EnhancedGCNLayer() for _ in range(args.gcn_layer)
                ])
        elif getattr(args, 'use_sage', False):
            print(f"🧠 Using GraphSAGE with aggregator: {getattr(args, 'sage_aggregator', 'mean')}")
            self.gcnLayers = nn.ModuleList([
                GraphSAGELayer(aggregator_type=getattr(args, 'sage_aggregator', 'mean')) 
                for _ in range(args.gcn_layer)
            ])
        else:
            print("📊 Using standard GCN layers")
            self.gcnLayers = nn.ModuleList([GCNLayer() for _ in range(args.gcn_layer)])
        
        self.gtLayers = ResidualGTLayer()
        self.pnnLayers = nn.Sequential(*[PNNLayer() for i in range(args.pnn_layer)])

    def getEgoEmbeds(self):
        return t.cat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, handler, is_test, sub, cmp, encoderAdj, decoderAdj=None):
        embeds = t.cat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        
        # GT layers for sub and cmp
        emb, _ = self.gtLayers(cmp, embeds)
        cList = [embeds, args.gtw*emb]
        emb, _ = self.gtLayers(sub, embeds)
        subList = [embeds, args.gtw*emb]

        # Apply GCN/UltraGCN/GraphSAGE layers with error handling
        for i, layer in enumerate(self.gcnLayers):
            try:
                embeds = layer(encoderAdj, embedsLst[-1])
                embeds2 = layer(sub, embedsLst[-1])
                embeds3 = layer(cmp, embedsLst[-1])
                subList.append(embeds2)
                embedsLst.append(embeds)
                cList.append(embeds3)
            except Exception as e:
                print(f"Layer {i} error: {e}, using fallback")
                # Fallback to simple GCN
                embeds = torch.spmm(encoderAdj, embedsLst[-1])
                embeds2 = torch.spmm(sub, embedsLst[-1])
                embeds3 = torch.spmm(cmp, embedsLst[-1])
                subList.append(embeds2)
                embedsLst.append(embeds)
                cList.append(embeds3)
        
        # Rest unchanged...
        if is_test is False:
            for i, pnn in enumerate(self.pnnLayers):
                embeds = pnn(handler, embedsLst[-1])
                embedsLst.append(embeds)
                
        if decoderAdj is not None:
            embeds, _ = self.gtLayers(decoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
            
        embeds = sum(embedsLst)
        cList = sum(cList)
        subList = sum(subList)

        return embeds[:args.user], embeds[args.user:], cList, subList





# class Model(nn.Module):
#     def __init__(self, gtLayer):
#         super(Model, self).__init__()

#         self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
#         self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        
#         # Use original GCN layers (this was working)
#         self.gcnLayer = GCNLayer()
        
#         self.gtLayers = ResidualGTLayer()
#         self.pnnLayers = nn.Sequential(*[PNNLayer() for i in range(args.pnn_layer)])

#     def getEgoEmbeds(self):
#         return t.cat([self.uEmbeds, self.iEmbeds], axis=0)

#     def forward(self, handler, is_test, sub, cmp, encoderAdj, decoderAdj=None):
#         embeds = t.cat([self.uEmbeds, self.iEmbeds], axis=0)
#         embedsLst = [embeds]
        
#         # GT layers for sub and cmp
#         emb, _ = self.gtLayers(cmp, embeds)
#         cList = [embeds, args.gtw*emb]
#         emb, _ = self.gtLayers(sub, embeds)
#         subList = [embeds, args.gtw*emb]

#         # Use original GCN layers (revert to working version)
#         for i in range(args.gcn_layer):
#             embeds = self.gcnLayer(encoderAdj, embedsLst[-1])
#             embeds2 = self.gcnLayer(sub, embedsLst[-1])
#             embeds3 = self.gcnLayer(cmp, embedsLst[-1])
#             subList.append(embeds2)
#             embedsLst.append(embeds)
#             cList.append(embeds3)
            
#         # PNN layers
#         if is_test is False:
#             for i, pnn in enumerate(self.pnnLayers):
#                 embeds = pnn(handler, embedsLst[-1])
#                 embedsLst.append(embeds)
                
#         # Decoder GT layer
#         if decoderAdj is not None:
#             embeds, _ = self.gtLayers(decoderAdj, embedsLst[-1])
#             embedsLst.append(embeds)
            
#         embeds = sum(embedsLst)
#         cList = sum(cList)
#         subList = sum(subList)

#         return embeds[:args.user], embeds[args.user:], cList, subList


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

# Remove the GraphSAGELayer class entirely and keep the rest of your classes as they were
class PNNLayer(nn.Module):
    def __init__(self):
        super(PNNLayer, self).__init__()
        self.linear_out_position = nn.Linear(args.latdim, 1)
        self.linear_out = nn.Linear(args.latdim, args.latdim)
        self.linear_hidden = nn.Linear(2 * args.latdim, args.latdim)
        self.act = nn.ReLU()

    def forward(self, handler, embeds):
        t.cuda.empty_cache()
        anchor_set_id = handler.anchorset_id
        dists_array = t.tensor(handler.dists_array, dtype=t.float32).to("cuda:0")
        set_ids_emb = embeds[anchor_set_id]
        set_ids_reshape = set_ids_emb.repeat(dists_array.shape[1], 1).reshape(-1, len(set_ids_emb),
                                                                              args.latdim)
        dists_array_emb = dists_array.T.unsqueeze(2)
        messages = set_ids_reshape * dists_array_emb

        self_feature = embeds.repeat(args.anchor_set_num, 1).reshape(-1, args.anchor_set_num, args.latdim)
        messages = torch.cat((messages, self_feature), dim=-1)
        messages = self.linear_hidden(messages).squeeze()

        outposition1 = t.mean(messages, dim=1)

        return outposition1

# Keep the rest of your classes unchanged...
class ResidualGTLayer(nn.Module):
    def __init__(self):
        super(ResidualGTLayer, self).__init__()
        self.gtLayer = GTLayer()

    def forward(self, adj, embeds, flag=False):
        x, att1 = self.gtLayer(adj, embeds, flag)
        y, att2 = self.gtLayer(adj, x, flag)
        y = y + x
        return y, (att1, att2)

class GTLayer(nn.Module):
    def __init__(self, dropout=0.):
        super(GTLayer, self).__init__()
     
        self.dropout = dropout

        self.head_dim = args.latdim // args.head
        assert self.head_dim * args.head == args.latdim

        self.qTrans = nn.Linear(args.latdim, args.latdim)
        self.kTrans = nn.Linear(args.latdim, args.latdim)
        self.vTrans = nn.Linear(args.latdim, args.latdim)

        self.out_proj = nn.Linear(args.latdim, args.latdim)

    def forward(self, adj, embeds, key_padding_mask=None, attn_mask=None):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]
        qEmbeds = self.qTrans(rowEmbeds).view([-1, args.head, args.latdim // args.head])
        kEmbeds = self.kTrans(colEmbeds).view([-1, args.head, args.latdim // args.head])
        vEmbeds = self.vTrans(colEmbeds).view([-1, args.head, args.latdim // args.head])

        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        tem = torch.zeros([adj.shape[0], args.head]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = torch.zeros([adj.shape[0], args.latdim]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)

        resEmbeds = self.out_proj(resEmbeds)

        return resEmbeds, att

class LocalGraph(nn.Module):
    def __init__(self, gtLayer):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer
        self.sft = t.nn.Softmax(0)
        self.device = "cuda:0"
        self.num_users = args.user
        self.num_items = args.item
        self.pnn = PNNLayer().cuda()

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff=None):
      dists_dict = {}
      for node in node_range:
        dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff)
      return dists_dict

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)
        num_workers = 1
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)

        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def precompute_dist_data(self, edge_index, num_nodes, approximate=0):
        graph = nx.Graph()
        graph.add_edges_from(edge_index)

        n = num_nodes
        dists_dict = self.all_pairs_shortest_path_length_parallel(graph,
                                                                  cutoff=approximate if approximate > 0 else None)
        dists_array = np.zeros((n, n), dtype=np.int8)

        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array

    def forward(self, adj, embeds, handler):
        embeds = self.pnn(handler, embeds)
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        tmp_rows = np.random.choice(rows.cpu(), size=[int(len(rows) * args.addRate)])
        tmp_cols = np.random.choice(cols.cpu(), size=[int(len(cols) * args.addRate)])

        add_cols = t.tensor(tmp_cols).to(self.device)
        add_rows = t.tensor(tmp_rows).to(self.device)

        newRows = t.cat([add_rows, add_cols, t.arange(args.user + args.item).cuda(), rows])
        newCols = t.cat([add_cols, add_rows, t.arange(args.user + args.item).cuda(), cols])

        ratings_keep = torch.ones(len(newRows), dtype=torch.float32)
        adj_mat = sp.csr_matrix((ratings_keep.cpu().numpy(), (newRows.cpu().numpy(), newCols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        embeds_l2, atten = self.gt_layer(add_adj, embeds)
        att_edge = t.sum(atten, dim=-1)

        return att_edge, add_adj


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.device = "cuda:0"
        self.sft = t.nn.Softmax(1)

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def create_sub_adj(self, adj, att_edge, flag):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        if flag:
            att_edge = (np.array(att_edge.detach().cpu() + 0.001))
        else:
            att_f = att_edge
            att_f[att_f > 3] = 3
            att_edge = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        
        att_f = att_edge / (att_edge.sum() + 1e-8)
        
        if np.isnan(att_f).any() or np.isinf(att_f).any():
            print("Warning: NaN/Inf detected in attention weights, using uniform distribution")
            att_f = np.ones(len(att_f)) / len(att_f)
        
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.sub),
                                      replace=False, p=att_f)

        keep_index.sort()

        drop_edges = []
        i = 0
        j = 0
        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(args.user + args.item).cuda(), rows])
        cols = t.cat([t.arange(args.user + args.item).cuda(), cols])

        ratings_keep = torch.ones(len(rows), dtype=torch.float32)
        adj_mat = sp.csr_matrix((ratings_keep.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = att_edge
        att_f[att_f > 3] = 3
        att_f = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f1 = att_f / (att_f.sum() + 1e-8)

        if np.isnan(att_f1).any() or np.isinf(att_f1).any():
            print("Warning: NaN/Inf detected in attention weights, using uniform distribution")
            att_f1 = np.ones(len(att_f1)) / len(att_f1)

        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.keepRate),
                                          replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(args.user + args.item).cuda(), rows])
        cols = t.cat([t.arange(args.user + args.item).cuda(), cols])
        drop_edges = []
        i, j = 0, 0

        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        ratings_keep = torch.ones(len(rows), dtype=torch.float32)
        adj_mat = sp.csr_matrix((ratings_keep.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        drop_row_ids = users_up[drop_edges]
        drop_col_ids = items_up[drop_edges]

        ext_rows = np.random.choice(rows.cpu(), size=[int(len(drop_row_ids) * args.ext)])
        ext_cols = np.random.choice(cols.cpu(), size=[int(len(drop_col_ids) * args.ext)])

        ext_cols = t.tensor(ext_cols).to(self.device)
        ext_rows = t.tensor(ext_rows).to(self.device)

        tmp_rows = t.cat([ext_rows, drop_row_ids])
        tmp_cols = t.cat([ext_cols, drop_col_ids])

        new_rows = np.random.choice(tmp_rows.cpu(), size=[int(adj._values().shape[0] * args.reRate)])
        new_cols = np.random.choice(tmp_cols.cpu(), size=[int(adj._values().shape[0] * args.reRate)])

        new_rows = t.tensor(new_rows).to(self.device)
        new_cols = t.tensor(new_cols).to(self.device)

        newRows = t.cat([new_rows, new_cols, t.arange(args.user + args.item).cuda(), rows])
        newCols = t.cat([new_cols, new_rows, t.arange(args.user + args.item).cuda(), cols])

        hashVal = newRows * (args.user + args.item) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (args.user + args.item)
        newRows = ((hashVal - newCols) / (args.user + args.item)).long()

        decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
                                          adj.shape)

        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp