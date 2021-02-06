echo "Transformer"
kedro run --params "{
  'dataset_parameters': {'flatten': true},
  'model_type': 'Transformer',
  'generic_model_params': {
    'd_model': 72,
    'num_heads': 9,
    'N': 6,
    'use_cuda': true,
    'embed_mode': 'Embedding',
    'dropout': 0.2
  },
  'optim_params': {
    'lr': 0.0001,
    'betas': [0.9, 0.999],
    'eps': 1e-9,
    'quantiles': [0.1, 0.5, 0.9],
    'warmup': false,
    'T_max_epoch': 100,
    'lr_min': 0.0001
  }
}"

echo "VanillaLSTM"
kedro run --params "{
  'dataset_parameters': {'flatten': true},
  'model_type': 'VanillaLSTM',
  'generic_model_params': {
    'linear_output_size': 72,
    'lstm_hidden_size': 72,
    'lstm_num_layers': 6,
    'hidden_seq_len': 10,
    'one_step_ahead': true,
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
    'lr_min': 0.0001
  }
}"

echo "VanillaLSTMSeq2Seq"
kedro run --params "{
  'dataset_parameters': {'flatten': true},
  'model_type': 'VanillaLSTMSeq2Seq',
  'generic_model_params': {
    'linear_output_size': 72,
    'lstm_hidden_size': 72,
    'lstm_num_layers': 6,
    'one_step_ahead': true,
    'use_cuda': true,
    'dropout': 0.2,
    'sigmoid_output': false
  },
  'optim_params': {
    'lr': 0.0001,
    'betas': [0.9, 0.999],
    'eps': 1e-9,
    'quantiles': [0.1, 0.5, 0.9],
    'warmup': false,
    'T_max_epoch': 100,
    'lr_min': 0.0001
  }
}"

echo "VanillaLSTMSeq2SeqAttn"
kedro run --params "{
  'dataset_parameters': {'flatten': true},
  'model_type': 'VanillaLSTMSeq2SeqAttn',
  'generic_model_params': {
    'linear_output_size': 72,
    'lstm_hidden_size': 72,
    'lstm_num_layers': 6,
    'one_step_ahead': true,
    'use_cuda': true,
    'dropout': 0.2,
    'sigmoid_output': false
  },
  'optim_params': {
    'lr': 0.0001,
    'betas': [0.9, 0.999],
    'eps': 1e-9,
    'quantiles': [0.1, 0.5, 0.9],
    'warmup': false,
    'T_max_epoch': 100,
    'lr_min': 0.0001
  }
}"

echo "ConvLSTM"
kedro run --params "{
  'dataset_parameters': {'flatten': false},
  'model_type': 'ConvLSTM',
  'generic_model_params': {
    'input_channels': 1,
    'hidden_channels': [8, 8, 8, 8, 8, 8],
    'kernel_size': 3,
    'pred_input_dim': 10,
    'one_step_ahead': true,
    'use_cuda': true,
    'dropout': 0.2,
    'sigmoid_output': false
  },
  'optim_params': {
    'lr': 0.0001,
    'betas': [0.9, 0.999],
    'eps': 1e-9,
    'quantiles': [0.1, 0.5, 0.9],
    'warmup': false,
    'T_max_epoch': 100,
    'lr_min': 0.0001
  }
}"