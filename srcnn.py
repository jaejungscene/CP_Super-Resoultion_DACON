import torch.nn as nn
import math

class SRCNN(nn.Module):
    def __init__(self, num_channels=3, feature_dim=64, map_dim=32):
        super(SRCNN, self).__init__()
        ## Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, feature_dim, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(True)
        )   # out = (2048, 2048)

        ## Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(feature_dim, map_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(True)
        )   # out = (2048, 2048)

        ## Rebuild the layer.
        self.reconstruction = nn.Conv2d(map_dim, num_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # out = (2048, 2048)

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


model_config = {
    #srcnn
    "srcnn": SRCNN(),
}

def create_model(model_name):
    return nn.DataParallel(model_config[model_name]).cuda()