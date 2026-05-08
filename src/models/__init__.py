from .decoder_only import DecoderOnlyAdder
from .carrier_encoder import InterleavedEncoderAdder

def get_model(config):
    m_type = config['model']['type']
    if m_type == 'interleaved':
        return DecoderOnlyAdder(config)
    elif m_type == 'carrier':
        return InterleavedEncoderAdder(config)
    else:
        raise ValueError(f"Unknown model type: {m_type}")