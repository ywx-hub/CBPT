from .cbpt import CBPT

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'cbpt':
        model = CBPT(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.CBPT.PATCH_SIZE,
                                in_chans=config.MODEL.CBPT.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.CBPT.EMBED_DIM,
                                depths=config.MODEL.CBPT.DEPTHS,
                                num_heads=config.MODEL.CBPT.NUM_HEADS,
                                window_size=config.MODEL.CBPT.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.CBPT.MLP_RATIO,
                                qkv_bias=config.MODEL.CBPT.QKV_BIAS,
                                qk_scale=config.MODEL.CBPT.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.CBPT.APE,
                                patch_norm=config.MODEL.CBPT.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
