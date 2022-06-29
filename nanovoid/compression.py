import torch


def compress_deltas(projector,batch_pfields):
    all_prod = None
    total_frame, total_pf, *tmp = batch_pfields.size()
    
    # print("projector",projector.dtype,projector.device)
    # print("pfields",batch_pfields.dtype,batch_pfields.device)
    
    for pfs in batch_pfields:
        pf_proj = None
        
        for g in range(total_pf):
            
            flat_vec = pfs[g].clone().flatten()
            prod = torch.matmul(projector,flat_vec) # for dense matrix from torch.rand
            
            # prod = projector.dot(flat_vec) # for sparse coo matrix from scipy.sparse
            # prod = torch.tensor(prod)
            
            prod = prod.unsqueeze(0)
            if pf_proj is None:
                pf_proj = prod
            else:
                pf_proj = torch.cat((pf_proj,prod))
        pf_proj = pf_proj.unsqueeze(0)
        if all_prod is None:
            all_prod = pf_proj
        else:
            all_prod = torch.cat((all_prod,pf_proj),0)
    # print("original_deltas.size=",batch_pfields.size())
    # print("projected_deltas.size=",all_prod.size())
    return all_prod



def compress_features(projector,batch_features):
    all_prod = None
    total_frames, total_pf = 3
    n_features = 15
    for features in batch_features:
        grain_feat_proj = None
        for g in range(total_pf):
            feat_proj = None
            for f in range(n_features):
            
                flat_vec = features[g][f].clone().flatten()
                prod = torch.matmul(projector,flat_vec)
                
                # prod = projector.dot(flat_vec) # for sparse coo matrix from scipy.sparse
                # prod = torch.tensor(prod)
                
                prod = prod.unsqueeze(0)
                if feat_proj is None:
                    feat_proj = prod
                else:
                    feat_proj = torch.cat((feat_proj,prod))
            
            feat_proj = feat_proj.unsqueeze(0)
            if grain_feat_proj is None:
                grain_feat_proj = feat_proj
            else:
                grain_feat_proj = torch.cat((grain_feat_proj,feat_proj))
        
        grain_feat_proj = grain_feat_proj.unsqueeze(0)
        if all_prod is None:
            all_prod = grain_feat_proj
        else:
            all_prod = torch.cat((all_prod,grain_feat_proj),0)
    # print("original.size=",batch_features.size())
    # print("projection.size=",all_prod.size())
    return all_prod