import numpy as np
import scipy.io as scio

from tqdm.auto import tqdm

from . import rank_metrics as MET

def calc_hammingDist(B1, B2):
    """B1 and B2 are sign vectors"""
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1,B2.transpose()))
    return distH

def one_hot_label(single_label, num_label=None):
    if num_label is None:
        num_label = np.max(single_label)+1
    num_samples = single_label.size
    one_hot_label = np.zeros([num_samples, num_label], int)
    for i in range(num_samples):
        one_hot_label[i, single_label[i]] = 1
    return one_hot_label

def precision_recall_curve(database_code, database_labels, validation_code, validation_labels, dist_type='hamming'):
    """Calculate precision
    code is vector of -1 and 1
    labels is OHE
    """
#     assert set(np.unique(database_code).tolist()) == set([-1, 1])
#     assert set(np.unique(validation_code).tolist()) == set([-1, 1])
#     assert len(database_labels.shape) == 2
#     assert len(validation_labels.shape) == 2
    
#     db_num = database_code.shape[0]
#     query_num = validation_code.shape[0]
        
#     if dist_type == 'hamming':
#         dist = calc_hammingDist(database_code, validation_code)
#         ids = np.argsort(dist, axis=0)
#     elif dist_type == 'cosine':
#         sim = np.dot(database_code, validation_code.T)
#         ids = np.argsort(-sim, axis=0)
#     else:
#         raise Exception('Unsupported distance type: {}'.format(dist_type))
    
#     APx = []
#     ARx = []
#     for i in tqdm(range(query_num)):
#         label = validation_labels[i]
#         idx = ids[:, i]
#         imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        
#         relevant_num = np.sum(imatch)
#         Lx = np.cumsum(imatch)
#         Px = Lx.astype(float) / np.arange(1, db_num+1, 1)
#         Rx = Lx.astype(float) / relevant_num
#         APx.append(Px)
#         ARx.append(Rx)
#     return np.mean(np.asarray(APx), axis=0), np.mean(np.asarray(ARx), axis=0)

    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
    assert len(database_labels.shape) == 2
    assert len(validation_labels.shape) == 2
    
    db_num = database_code.shape[0]
    query_num = validation_code.shape[0]

    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    APx = []
    ARx = []
    
    for i in tqdm(range(query_num)):
        label = validation_labels[i]
        idx = ids[:, i]
        imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        
        Rx = Lx / relevant_num
        Px = Lx / np.arange(1, db_num+1, 1)
        
        ARx.append(Rx)
        APx.append(Px)
        
    APx = np.vstack(APx)
    ARx = np.vstack(ARx)

    return np.mean(APx, axis=0), np.mean(ARx, axis=0)

def precision_curve(database_code, database_labels, validation_code, validation_labels, max_R, dist_type='hamming'):
    """Calculate precision curve at various thresholds"""
    
    return precision(database_code, database_labels, validation_code, validation_labels, range(1, max_R + 1), dist_type='hamming')

def precision(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
    """Calculate precision
    code is vector of -1 and 1
    labels is OHE
    """
    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
    assert len(database_labels.shape) == 2
    assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]
        
    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine':
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    
    APx = {R: [] for R in Rs}

    for i in tqdm(range(query_num)):
        label = validation_labels[i]
        idx = ids[:, i]
        imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        for R in Rs:
            relevant_num = np.sum(imatch[:R])
            if relevant_num != 0:
                APx[R].append(float(relevant_num) / R)
    
    #Compute 2 types of precisions: one ignores 0-relevant and one includes 0-relevant
    return {R: (np.mean(np.array(APxR)), np.sum(np.array(APxR)) / query_num) for (R, APxR) in APx.items()}

# def mean_average_precision(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
#     """Compute mAP
#     code is vector of -1 and 1
#     labels is OHE
#     """
#     assert set(np.unique(database_code).tolist()) == set([-1, 1])
#     assert set(np.unique(validation_code).tolist()) == set([-1, 1])
#     assert len(database_labels.shape) == 2
#     assert len(validation_labels.shape) == 2
    
#     query_num, db_num = validation_code.shape[0], database_code.shape[0]

#     if dist_type == 'hamming':
#         dist = calc_hammingDist(database_code, validation_code)
#     elif dist_type == 'cosine':
#         sim = np.dot(database_code, validation_code.T)
#         dist = -sim
#     else:
#         raise Exception('Unsupported distance type: {}'.format(dist_type))
    
#     APx = []

#     for i in tqdm(range(query_num)):
#         label = validation_labels[i]
#         idx = np.lexsort((np.arange(db_num), dist[:, i]))
#         imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
#         Lx = np.cumsum(imatch)
#         R = Rs[0]
#         relevant_num = np.sum(imatch[:R])
#         Px = Lx[:R].astype(float) / np.arange(1, R+1, 1)
#         if relevant_num != 0:
#             APx.append((np.sum(Px * imatch[:R]) / relevant_num, relevant_num))

#     return APx

# def mean_average_precision(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
#     """Compute mAP
#     code is vector of -1 and 1
#     labels is OHE
#     """
#     assert set(np.unique(database_code).tolist()) == set([-1, 1])
#     assert set(np.unique(validation_code).tolist()) == set([-1, 1])
#     assert len(database_labels.shape) == 2
#     assert len(validation_labels.shape) == 2
    
#     query_num, db_num = validation_code.shape[0], database_code.shape[0]

#     if dist_type == 'hamming':
#         dist = calc_hammingDist(database_code, validation_code)
#     elif dist_type == 'cosine':
#         sim = np.dot(database_code, validation_code.T)
#         dist = -sim
#     else:
#         raise Exception('Unsupported distance type: {}'.format(dist_type))
    
#     APx = {R: [] for R in Rs}

#     for i in tqdm(range(query_num)):
#         label = validation_labels[i]
#         idx = np.lexsort((np.arange(db_num), dist[:, i]))
#         imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
#         Lx = np.cumsum(imatch)
#         for R in Rs:
#             relevant_num = np.sum(imatch[:R])
#             Px = Lx[:R].astype(float) / np.arange(1, R+1, 1)
#             if relevant_num != 0:
#                 APx[R].append(np.sum(Px * imatch[:R]) / relevant_num)

#     return {R: np.mean(np.array(APxR)) for (R, APxR) in APx.items()}

def mean_average_precision(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
    """Compute mAP
    code is vector of -1 and 1
    labels is OHE
    """
    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
    assert len(database_labels.shape) == 2
    assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]

    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    APx = {R: [] for R in Rs}

    for i in tqdm(range(query_num)):
        label = validation_labels[i]
        idx = ids[:, i]
        imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        Lx = np.cumsum(imatch)
        for R in Rs:
            relevant_num = np.sum(imatch[:R])
            Px = Lx[:R].astype(float) / np.arange(1, R+1, 1)
            if relevant_num != 0:
                APx[R].append(np.sum(Px * imatch[:R]) / relevant_num)
    
    #Compute 2 types of mAP: one ignores 0-relevant and one includes 0-relevant
    return {R: (np.mean(np.array(APxR)), np.sum(np.array(APxR)) / query_num) for (R, APxR) in APx.items()}

def calculate_distances(database_code, validation_code, dist_type='hamming'):
    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
#     assert len(database_labels.shape) == 2
#     assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]

    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        dist = -sim
        ids = np.argsort(dist, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    return dist, ids

def calculate_all_metrics(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming', strict=False):
    #Rs = [database_code.shape[0] if R == -1 else R for R in Rs]
    
    if strict:
        assert set(np.unique(database_code).tolist()) == set([-1, 1])
        assert set(np.unique(validation_code).tolist()) == set([-1, 1])
        assert len(database_labels.shape) == 2
        assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]

    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    mean_Px = {R: [] for R in Rs} #mean_precision
    mean_APx = {R: [] for R in Rs} #mean_average_precision
    
    for i in tqdm(range(query_num), desc='Ranking'):
        label = validation_labels[i]
        idx = ids[:, i]
        imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        Lx = np.cumsum(imatch)
        for R in Rs:
            relevant_num = np.sum(imatch[:R])
            Px = Lx[:R].astype(float) / np.arange(1, R+1, 1)
            if relevant_num != 0:
                mean_APx[R].append(np.sum(Px * imatch[:R]) / relevant_num)
                mean_Px[R].append(float(relevant_num) / R)

    mean_Px = {R: np.sum(np.array(xR)) / query_num for (R, xR) in mean_Px.items()} 
    mean_APx = {R: np.sum(np.array(xR)) / query_num for (R, xR) in mean_APx.items()} 
    return mean_Px, mean_APx

import concurrent.futures

def calculate_all_metrics_parallel(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
    assert len(database_labels.shape) == 2
    assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]

    print('Compute distance')
    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    mean_Px = {R: [] for R in Rs} #mean_precision
    mean_APx = {R: [] for R in Rs} #mean_average_precision
    
    def compute_result(Rs, label, ranked_items):
        imatch = (np.dot(ranked_items, label) > 0).astype(np.int)
        Lx = np.cumsum(imatch)
        APx_R = {}
        Px_R = {}
        for R in Rs:
            relevant_num = np.sum(imatch[:R])
            Px = Lx[:R].astype(float) / np.arange(1, R+1, 1)
            if relevant_num != 0:
                APx_R[R] = np.sum(Px * imatch[:R]) / relevant_num
                Px_R[R] = float(relevant_num) / R
            else:
                APx_R[R] = 0
                Px_R[R] = 0
        return (APx_R, Px_R)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        print('Submit query eval')
#         future_to_results = {}
#         for i in tqdm(range(query_num)):
#             label = validation_labels[i]
#             idx = ids[:, i]
#             ranked_items = database_labels[idx, :]
#             future_to_results[executor.submit(compute_result, Rs, label, ranked_items)] = i
            
        future_to_results = {executor.submit(compute_result, Rs, validation_labels[i], database_labels[ids[:, i], :]): i for i in range(query_num)}
        print('Get results')
        for future in tqdm(concurrent.futures.as_completed(future_to_results)):
            i = future_to_results[future]
            try:
                APx_R, Px_R = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (i, exc))
            else:
                for R, v in Px_R.item():
                    if v != 0:
                        mean_Px[R].append(v)
                for R, v in APx_R.item():
                    if v != 0:
                        mean_APx[R].append(v)

    mean_Px = {R: (np.mean(np.array(xR)), np.sum(np.array(xR)) / query_num) for (R, xR) in mean_Px.items()} 
    mean_APx = {R: (np.mean(np.array(xR)), np.sum(np.array(xR)) / query_num) for (R, xR) in mean_APx.items()} 
    return mean_Px, mean_APx

def calculate_all_metrics_v2(database_code, database_labels, validation_code, validation_labels, Rs, dist_type='hamming'):
    assert set(np.unique(database_code).tolist()) == set([-1, 1])
    assert set(np.unique(validation_code).tolist()) == set([-1, 1])
    assert len(database_labels.shape) == 2
    assert len(validation_labels.shape) == 2
    
    query_num = validation_code.shape[0]

    if dist_type == 'hamming':
        dist = calc_hammingDist(database_code, validation_code)
        ids = np.argsort(dist, axis=0)
    elif dist_type == 'cosine': 
        sim = np.dot(database_code, validation_code.T)
        ids = np.argsort(-sim, axis=0)
    else:
        raise Exception('Unsupported distance type: {}'.format(dist_type))
    
    mean_Px = {R: [] for R in Rs} #mean_precision
    mean_APx = {R: [] for R in Rs} #mean_average_precision

    
    for i in tqdm(range(query_num)):
        label = validation_labels[i]
        idx = ids[:, i]
        imatch = (np.dot(database_labels[idx, :], label) > 0).astype(np.int)
        
        for R in Rs:
            imatch_R = imatch[:R]
            relevant_num = np.sum(imatch_R)
            
            mean_Px[R].append(MET.precision_at_k(imatch_R, R))
            mean_APx[R].append(MET.average_precision(imatch_R))
    mean_Px = {R: np.mean(np.array(xR)) for (R, xR) in mean_Px.items()} 
    mean_APx = {R: np.mean(np.array(xR)) for (R, xR) in mean_APx.items()} 
    return mean_Px, mean_APx            
            
    
def unique_hash_codes(b):
    return np.unique(b, axis=0)
