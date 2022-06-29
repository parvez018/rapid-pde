import torch

def get_deltas(all_data):
    # input: batch_size list of dictionary items, each item in dictionary has 4 key-value pairs
            # 'step': integer, 'cv': grid, 'ci': grid, 'eta': grid
    # output: tensor batch_size x 3 x width x height
    #           field variables are ordered as cv,ci,eta in this order
    all_deltas = None
    total_frames = len(all_data)
    for f in range(total_frames-1):
        cv_delta = all_data[f+1]['cv'] - all_data[f]['cv']
        ci_delta = all_data[f+1]['ci'] - all_data[f]['ci']
        eta_delta = all_data[f+1]['eta'] - all_data[f]['eta']
        
        cv_delta = cv_delta.unsqueeze(0)
        ci_delta = ci_delta.unsqueeze(0)
        eta_delta = eta_delta.unsqueeze(0)
        
        deltas = torch.cat((cv_delta,ci_delta,eta_delta),0)
        deltas = deltas.unsqueeze(0)
        
        if all_deltas is None:
            all_deltas = deltas
        else:
            all_deltas = torch.cat((all_deltas,deltas))
    
    return all_deltas