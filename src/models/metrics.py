"""
Evaluation metrics for OpenSet recognition in biometric identification.

Implements metrics required for OpenSet evaluation:
- EER (Equal Error Rate)
- OSCR (Open-Set Classification Rate)
- AUROC (Area Under ROC curve for known vs unknown)
- TPR@FPR (True Positive Rate at specific False Positive Rate)
- FRR@FAR (False Rejection Rate at specific False Acceptance Rate)
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) for verification.
    
    EER is the point where FAR (False Acceptance Rate) equals FRR (False Rejection Rate).
    Lower EER indicates better performance.
    
    Args:
        genuine_scores: Similarity scores for genuine (same-class) pairs.
        impostor_scores: Similarity scores for impostor (different-class) pairs.
        
    Returns:
        Tuple of (eer, threshold) where:
        - eer: Equal Error Rate as a percentage (0-100)
        - threshold: Threshold value at EER point
        
    Note:
        For genuine scores: higher is better (more similar)
        For impostor scores: lower is better (less similar)
    """
    # Combine scores and create labels
    # Label 1 for genuine, 0 for impostor
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # FRR = 1 - TPR
    frr = 1 - tpr
    
    # Find where FAR == FRR
    eer_idx = np.nanargmin(np.absolute(fpr - frr))
    eer = (fpr[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return float(eer * 100), float(eer_threshold)


def compute_frr_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 0.01,
) -> Tuple[float, float]:
    """Compute False Rejection Rate at a target False Acceptance Rate.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs.
        impostor_scores: Similarity scores for impostor pairs.
        target_far: Target False Acceptance Rate (default 0.01 = 1%).
        
    Returns:
        Tuple of (frr, threshold) where:
        - frr: False Rejection Rate at target FAR (as percentage)
        - threshold: Threshold achieving the target FAR
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    frr = 1 - tpr
    
    # Find threshold closest to target FAR
    idx = np.nanargmin(np.absolute(fpr - target_far))
    
    return float(frr[idx] * 100), float(thresholds[idx])


def compute_tpr_at_fpr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_fpr: float = 0.001,
) -> Tuple[float, float]:
    """Compute True Positive Rate at a target False Positive Rate.
    
    Used for OpenSet evaluation: measures ability to accept known classes
    while rejecting unknown classes at a specific FPR.
    
    Args:
        known_scores: Maximum similarity scores for known class samples.
        unknown_scores: Maximum similarity scores for unknown class samples.
        target_fpr: Target False Positive Rate (default 0.001 = 0.1%).
        
    Returns:
        Tuple of (tpr, threshold) where:
        - tpr: True Positive Rate at target FPR (as percentage)
        - threshold: Threshold achieving the target FPR
        
    Note:
        Known samples should have high scores (accepted as known).
        Unknown samples should have low scores (rejected as unknown).
    """
    scores = np.concatenate([known_scores, unknown_scores])
    # Label 1 for known, 0 for unknown
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find threshold closest to target FPR
    idx = np.nanargmin(np.absolute(fpr - target_fpr))
    
    return float(tpr[idx] * 100), float(thresholds[idx])


def compute_auroc(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    """Compute Area Under ROC curve for known vs unknown discrimination.
    
    Measures the model's ability to distinguish between known and unknown classes.
    AUROC = 1.0 means perfect separation.
    AUROC = 0.5 means random guessing.
    
    Args:
        known_scores: Maximum similarity scores for known class samples.
        unknown_scores: Maximum similarity scores for unknown class samples.
        
    Returns:
        AUROC value in range [0, 1].
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])
    
    return float(roc_auc_score(labels, scores))


def compute_oscr(
    known_predictions: np.ndarray,
    known_labels: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    thresholds: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Compute Open-Set Classification Rate (OSCR) curve.
    
    OSCR measures correct classification rate (CCR) vs False Positive Rate (FPR).
    - CCR: Fraction of known samples correctly classified AND above threshold
    - FPR: Fraction of unknown samples incorrectly accepted as known
    
    Args:
        known_predictions: Predicted class labels for known samples (0-indexed).
        known_labels: True class labels for known samples (0-indexed).
        known_scores: Maximum similarity scores for known samples.
        unknown_scores: Maximum similarity scores for unknown samples.
        thresholds: Array of thresholds to evaluate (if None, auto-generated).
        
    Returns:
        Dictionary with:
        - 'ccr': Correct Classification Rate at each threshold
        - 'fpr': False Positive Rate at each threshold
        - 'thresholds': Threshold values
        - 'oscr_auc': Area under OSCR curve
    """
    if thresholds is None:
        # Generate thresholds from score range
        all_scores = np.concatenate([known_scores, unknown_scores])
        thresholds = np.linspace(
            all_scores.min() - 0.1,
            all_scores.max() + 0.1,
            200
        )
    
    ccr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        # CCR: fraction of known samples correctly classified AND above threshold
        correct_and_accepted = (
            (known_predictions == known_labels) &
            (known_scores >= threshold)
        )
        ccr = correct_and_accepted.sum() / len(known_labels)
        ccr_list.append(ccr)
        
        # FPR: fraction of unknown samples accepted (above threshold)
        unknown_accepted = (unknown_scores >= threshold).sum()
        fpr = unknown_accepted / len(unknown_scores)
        fpr_list.append(fpr)
    
    ccr_array = np.array(ccr_list)
    fpr_array = np.array(fpr_list)
    
    # Compute AUC of OSCR curve
    # Sort by FPR for AUC computation
    sorted_indices = np.argsort(fpr_array)
    oscr_auc = auc(fpr_array[sorted_indices], ccr_array[sorted_indices])
    
    return {
        'ccr': ccr_array,
        'fpr': fpr_array,
        'thresholds': thresholds,
        'oscr_auc': float(oscr_auc),
    }


def compute_openset_metrics(
    known_predictions: np.ndarray,
    known_labels: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive OpenSet evaluation metrics.
    
    Combines all important metrics for OpenSet recognition evaluation.
    
    Args:
        known_predictions: Predicted class labels for known samples.
        known_labels: True class labels for known samples.
        known_scores: Maximum similarity scores for known samples.
        unknown_scores: Maximum similarity scores for unknown samples.
        
    Returns:
        Dictionary with all computed metrics:
        - 'auroc': Area under ROC for known vs unknown
        - 'eer': Equal Error Rate (percentage)
        - 'eer_threshold': Threshold at EER
        - 'tpr_at_fpr_0.001': TPR at 0.1% FPR
        - 'tpr_threshold_0.001': Threshold for TPR@0.1%FPR
        - 'oscr_auc': Area under OSCR curve
        - 'accuracy_known': Classification accuracy on known samples
    """
    metrics = {}
    
    # Handle edge case: no unknown samples (closed-set evaluation)
    if len(unknown_scores) == 0:
        metrics['auroc'] = float('nan')
        metrics['eer'] = float('nan')
        metrics['eer_threshold'] = float('nan')
        metrics['tpr_at_fpr_0.001'] = float('nan')
        metrics['tpr_threshold_0.001'] = float('nan')
        metrics['tpr_at_fpr_0.01'] = float('nan')
        metrics['tpr_threshold_0.01'] = float('nan')
        metrics['tpr_at_fpr_0.1'] = float('nan')
        metrics['tpr_threshold_0.1'] = float('nan')
        metrics['oscr_auc'] = float('nan')
        metrics['oscr'] = float('nan')
    else:
        # AUROC for known vs unknown discrimination
        metrics['auroc'] = compute_auroc(known_scores, unknown_scores)
        
        # EER (treating unknown as impostor)
        # For this, we need genuine and impostor scores
        # Use known scores as genuine, unknown as impostor
        eer, eer_threshold = compute_eer(known_scores, unknown_scores)
        metrics['eer'] = eer
        metrics['eer_threshold'] = eer_threshold
        
        # TPR at FPR=0.001 (0.1%)
        tpr_001, tpr_threshold_001 = compute_tpr_at_fpr(known_scores, unknown_scores, target_fpr=0.001)
        metrics['tpr_at_fpr_0.001'] = tpr_001
        metrics['tpr_threshold_0.001'] = tpr_threshold_001
        
        # TPR at FPR=0.01 (1%)
        tpr_01, tpr_threshold_01 = compute_tpr_at_fpr(known_scores, unknown_scores, target_fpr=0.01)
        metrics['tpr_at_fpr_0.01'] = tpr_01
        metrics['tpr_threshold_0.01'] = tpr_threshold_01
        
        # TPR at FPR=0.1 (10%)
        tpr_1, tpr_threshold_1 = compute_tpr_at_fpr(known_scores, unknown_scores, target_fpr=0.1)
        metrics['tpr_at_fpr_0.1'] = tpr_1
        metrics['tpr_threshold_0.1'] = tpr_threshold_1
        
        # OSCR
        oscr_result = compute_oscr(
            known_predictions,
            known_labels,
            known_scores,
            unknown_scores
        )
        metrics['oscr_auc'] = oscr_result['oscr_auc']
        metrics['oscr'] = oscr_result['oscr_auc']  # Alias for convenience
    
    # Classification accuracy on known samples (closed-set)
    known_predictions = np.asarray(known_predictions)
    known_labels = np.asarray(known_labels)
    accuracy = (known_predictions == known_labels).mean()
    metrics['accuracy_known'] = float(accuracy * 100)
    
    # Aliases for convenience (matching test expectations)
    metrics['tpr_at_fpr_001'] = metrics['tpr_at_fpr_0.001']
    metrics['tpr_at_fpr_01'] = metrics['tpr_at_fpr_0.1']
    
    return metrics


def compute_cmc_curve(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 20,
) -> np.ndarray:
    """Compute Cumulative Match Characteristic (CMC) curve for identification.
    
    CMC curve shows the probability that the correct match is within the top-k
    retrieved results. Used for 1:N identification tasks.
    
    Args:
        query_embeddings: Query embedding vectors (N_query, embedding_dim).
        gallery_embeddings: Gallery embedding vectors (N_gallery, embedding_dim).
        query_labels: True labels for query samples (N_query,).
        gallery_labels: True labels for gallery samples (N_gallery,).
        max_rank: Maximum rank to compute (default 20).
        
    Returns:
        CMC curve: array of shape (max_rank,) with recognition rate at each rank.
        
    Note:
        Rank-1 = recognition rate when only top-1 match is considered.
        Rank-k = recognition rate when top-k matches are considered.
    """
    n_queries = len(query_embeddings)
    cmc = np.zeros(max_rank)
    
    for i in range(n_queries):
        query_emb = query_embeddings[i:i+1]  # (1, embedding_dim)
        query_label = query_labels[i]
        
        # Compute cosine similarities (embeddings assumed L2-normalized)
        similarities = np.dot(query_emb, gallery_embeddings.T)[0]  # (N_gallery,)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarities)
        sorted_labels = gallery_labels[sorted_indices]
        
        # Find rank of first correct match
        correct_matches = (sorted_labels == query_label)
        if correct_matches.any():
            first_match_rank = np.where(correct_matches)[0][0]
            
            # Increment CMC for all ranks >= first match rank
            if first_match_rank < max_rank:
                cmc[first_match_rank:] += 1
    
    # Normalize by number of queries
    cmc = cmc / n_queries
    
    return cmc
