results_data = {
    'Model': [
        'TF-IDF (with images)', 
        'TF-IDF (no images)',
        'Matrix Factorization (with images)',
        'Matrix Factorization (no images)',
        'BLaIR-CLIP-Unfrozen (text)',
        'BLaIR-CLIP-Unfrozen (image)',
        'BLaIR-CLIP-Unfrozen (combined)',
        'BLaIR-CLIP-Frozen (text)',
        'BLaIR-CLIP-Frozen (image)',
        'BLaIR-CLIP-Frozen (combined)',
        'Untrained BLaIR-CLIP (text)',
        'Untrained BLaIR-CLIP (image)',
        'Untrained BLaIR-CLIP (combined)'
    ],
    'Recall@10': [
        0.0139,  # TF-IDF with images
        0.0139,  # TF-IDF without images
        0.0064,  # MF with images
        0.0064,  # MF without images
        0.08354351344468088,   # BLaIR-CLIP-Unfrozen (text)
        0.04405156053597982,   # BLaIR-CLIP-Unfrozen (image)
        0.08298485881647374,   # BLaIR-CLIP-Unfrozen (combined)
        0.08351475916234667,   # BLaIR-CLIP-Frozen (text)
        0.06330460643602993,   # BLaIR-CLIP-Frozen (image)
        0.08166626958372014,   # BLaIR-CLIP-Frozen (combined)
        0.0724,                # Untrained BLaIR-CLIP (text)
        0.0641,                # Untrained BLaIR-CLIP (image)
        0.0640                 # Untrained BLaIR-CLIP (combined)
    ],
    'Recall@50': [
        0.0376,  # TF-IDF with images
        0.0376,  # TF-IDF without images
        0.0203,  # MF with images
        0.0203,  # MF without images
        0.12904100360660856,   # BLaIR-CLIP-Unfrozen (text)
        0.08881376262107607,   # BLaIR-CLIP-Unfrozen (image)
        0.12847824122378226,   # BLaIR-CLIP-Unfrozen (combined)
        0.1289259864772718,    # BLaIR-CLIP-Frozen (text)
        0.08334634122296071,   # BLaIR-CLIP-Frozen (image)
        0.12146630408885895,   # BLaIR-CLIP-Frozen (combined)
        0.1028,                # Untrained BLaIR-CLIP (text)
        0.0832,                # Untrained BLaIR-CLIP (image)
        0.0831                 # Untrained BLaIR-CLIP (combined)
    ],
    'NDCG@10': [
        None,  # TF-IDF with images - not provided
        None,  # TF-IDF without images - not provided
        None,  # MF with images - not provided
        None,  # MF without images - not provided
        0.06818774239243472,   # BLaIR-CLIP-Unfrozen (text)
        0.04116633266546048,   # BLaIR-CLIP-Unfrozen (image)
        0.06691807487858262,   # BLaIR-CLIP-Unfrozen (combined)
        0.06818229992221596,   # BLaIR-CLIP-Frozen (text)
        0.05596700403525989,   # BLaIR-CLIP-Frozen (image)
        0.06723843841463854,   # BLaIR-CLIP-Frozen (combined)
        0.0596,                # Untrained BLaIR-CLIP (text)
        0.0566,                # Untrained BLaIR-CLIP (image)
        0.0567                 # Untrained BLaIR-CLIP (combined)
    ],
    'MRR': [
        None,  # TF-IDF with images - not provided
        None,  # TF-IDF without images - not provided
        None,  # MF with images - not provided
        None,  # MF without images - not provided
        0.0666833593669148,    # BLaIR-CLIP-Unfrozen (text)
        0.04215245177913969,   # BLaIR-CLIP-Unfrozen (image)
        0.06515428035562067,   # BLaIR-CLIP-Unfrozen (combined)
        0.0666756377334156,    # BLaIR-CLIP-Frozen (text)
        0.05515796272445119,   # BLaIR-CLIP-Frozen (image)
        0.06561160075607803,   # BLaIR-CLIP-Frozen (combined)
        0.0578,                # Untrained BLaIR-CLIP (text)
        0.0557,                # Untrained BLaIR-CLIP (image)
        0.0558                 # Untrained BLaIR-CLIP (combined)
    ],
    'Uses Images': [
        'No (text only)',
        'No (text only)', 
        'No (interactions only)',
        'No (interactions only)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)',
        'Yes (multi-modal)'
    ]
}