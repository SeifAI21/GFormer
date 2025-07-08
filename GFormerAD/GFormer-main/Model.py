# Add this new GraphSAGE layer class
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, aggregator_type='mean', dropout=0.1):
        super(GraphSAGELayer, self).__init__()
        
        self.in_dim = in_dim or args.latdim
        self.out_dim = out_dim or args.latdim
        self.aggregator_type = aggregator_type
        self.dropout = nn.Dropout(dropout)
        
        # Linear transformations
        self.self_linear = nn.Linear(self.in_dim, self.out_dim)
        self.neigh_linear = nn.Linear(self.in_dim, self.out_dim)
        
        # Aggregator-specific components
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self.in_dim, self.out_dim, batch_first=True)
        elif aggregator_type == 'pool':
            self.pool_linear = nn.Linear(self.in_dim, self.out_dim)
        
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(self.out_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neigh_linear.weight)

    def aggregate_neighbors(self, adj, embeds):
        """Aggregate neighbor embeddings using specified aggregator"""
        indices = adj._indices()
        values = adj._values()
        rows, cols = indices[0, :], indices[1, :]
        
        # Get neighbor embeddings
        neighbor_embeds = embeds[cols]
        
        if self.aggregator_type == 'mean':
            # Mean aggregation (most common and effective)
            neighbor_agg = torch.spmm(adj, embeds)
            
        elif self.aggregator_type == 'max':
            # Max pooling aggregation
            neighbor_agg = torch.zeros_like(embeds)
            for i in range(adj.shape[0]):
                mask = (rows == i)
                if mask.sum() > 0:
                    node_neighbors = neighbor_embeds[mask]
                    neighbor_agg[i] = torch.max(node_neighbors, dim=0)[0]
                else:
                    neighbor_agg[i] = embeds[i]  # Self if no neighbors
                    
        elif self.aggregator_type == 'lstm':
            # LSTM aggregation
            neighbor_agg = torch.zeros_like(embeds)
            for i in range(adj.shape[0]):
                mask = (rows == i)
                if mask.sum() > 0:
                    node_neighbors = neighbor_embeds[mask].unsqueeze(0)
                    lstm_out, _ = self.lstm(node_neighbors)
                    neighbor_agg[i] = lstm_out.squeeze(0)[-1]  # Last output
                else:
                    neighbor_agg[i] = embeds[i]
                    
        elif self.aggregator_type == 'pool':
            # Pooling aggregation
            neighbor_agg = torch.zeros_like(embeds)
            for i in range(adj.shape[0]):
                mask = (rows == i)
                if mask.sum() > 0:
                    node_neighbors = neighbor_embeds[mask]
                    pooled = torch.mean(self.activation(self.pool_linear(node_neighbors)), dim=0)
                    neighbor_agg[i] = pooled
                else:
                    neighbor_agg[i] = embeds[i]
        
        return neighbor_agg

    def forward(self, adj, embeds):
        # Apply dropout to input embeddings
        embeds = self.dropout(embeds)
        
        # Aggregate neighbors
        neighbor_agg = self.aggregate_neighbors(adj, embeds)
        
        # Transform self and neighbor embeddings
        self_transformed = self.self_linear(embeds)
        neigh_transformed = self.neigh_linear(neighbor_agg)
        
        # Combine self and neighbor information
        combined = self_transformed + neigh_transformed
        
        # Apply activation and normalization
        output = self.norm(self.activation(combined))
        
        return output

# Update your main Model class
class Model(nn.Module):
    def __init__(self, gtLayer):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        
        # REPLACED: GCN layers with GraphSAGE layers
        # You can experiment with different aggregators
        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(aggregator_type='mean') for _ in range(args.gcn_layer)
        ])
        
        # Keep one GCN layer for compatibility (optional)
        self.gcnLayer = GCNLayer()
        
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

        # REPLACED: Use GraphSAGE layers instead of GCN
        for i, sage_layer in enumerate(self.sage_layers):
            embeds = sage_layer(encoderAdj, embedsLst[-1])
            embeds2 = sage_layer(sub, embedsLst[-1])
            embeds3 = sage_layer(cmp, embedsLst[-1])
            subList.append(embeds2)
            embedsLst.append(embeds)
            cList.append(embeds3)
            
        # Rest remains the same
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