import timm

def get_backbone(name="resnet50", pretrained=True, in_chans=3):
    """Create a timm backbone without a classifier.

    Returns:
        model: feature extractor (num_classes=0).
        feat_dim: output feature dimensionality.
    """
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
    feat_dim = m.num_features   # embedding size of the backbone
    return m, feat_dim
