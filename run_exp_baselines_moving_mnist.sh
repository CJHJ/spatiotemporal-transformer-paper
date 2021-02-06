echo "Transformer"
kedro run --env=moving_mnist --pipeline=moving_mnist --params "{
  'dataset_parameters': {'flatten': true},
  'model_type': 'Transformer',
  'generic_model_params': {
    'd_model': 72,
    'num_heads': 9,
    'N': 6,
    'use_cuda': true,
    'embed_mode': 'Embedding',
    'dropout': 0.2
  }
}"

echo "VanillaLSTM"
kedro run --env=moving_mnist --pipeline=moving_mnist --params "{
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
  }
}"

echo "VanillaLSTMSeq2Seq"
kedro run --env=moving_mnist --pipeline=moving_mnist --params "{
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
  }
}"

echo "VanillaLSTMSeq2SeqAttn"
kedro run --env=moving_mnist --pipeline=moving_mnist --params "{
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
  }
}"

echo "ConvLSTM"
kedro run --env=moving_mnist --pipeline=moving_mnist --params "{
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
  }
}"