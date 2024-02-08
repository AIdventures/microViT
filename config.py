###################################################################################
################################## CONFIGURATION ##################################
###################################################################################
class ViTConfig:
    in_channels: int = 1  # MNIST images only have 1 channel
    img_size: int = 28  # MNIST images are 28x28
    patch_size: int = 7
    num_classes: int = 10  # MNIST number of classes
    n_encoder_layer: int = 3
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

VIT_CONFIG = ViTConfig
