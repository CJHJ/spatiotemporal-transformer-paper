echo "SpaTransSeq2SeqProbs"
kedro run --params "{
  'train_data_parameters': {
    'batch_size': 12,
    'shuffle': true,
    'num_workers': 4
  },
  'generic_model_params': {
    'patch_size': 8,
    'd_model': 72,
    'num_heads': 9,
    'num_layers': 6,
    'use_cuda': true,
    'dropout': 0.2
  },
  'optim_params': {
    'lr': 0.0001,
    'betas': [0.9, 0.999],
    'eps': 1e-9,
    'quantiles': [0.1, 0.5, 0.9],
    'warmup': false,
    'T_max_epoch': 100,
    'lr_min': 0.00001
  }
}"