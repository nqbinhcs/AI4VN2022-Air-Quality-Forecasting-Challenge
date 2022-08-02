

class BaseClassifier(nn.Module):
    def __init__(self, extractor_cfg, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.classifier = nn.Linear(self.feature_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor.get_embedding(x)
        return self.classifier(x)
