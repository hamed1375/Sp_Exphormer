from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    cfg.train.mode = 'custom_train'
    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False
    # If True, after training, it will rerun the model for saving the attention scores
    cfg.train.save_attention_scores = False
    cfg.train.saving_epoch = False  # don't change this, this is just for the model itself to adjust when to save
    cfg.train.cur_epoch = 0
    cfg.train.temp_rdc_ratio = 1.0
    cfg.train.temp_min = 0.1
    cfg.train.temp_wait = 10
    cfg.train.replace_edges = False
    cfg.train.number_of_edge_sets = 1
    cfg.train.rotate_edges = False
    cfg.train.layer_wise_edges = False
    cfg.train.sample_new_edges = False
    cfg.train.num_edge_samples = 50
    cfg.train.edge_sample_num_neighbors = [6, 4, 4, 4, 4, 4]
    cfg.train.spexphormer_sampler = 'graph_reservoir'
    cfg.train.resampling_epochs = 1
