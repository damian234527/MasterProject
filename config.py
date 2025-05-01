GENERAL_CONFIG = {
    "seed": 42

}

HEADLINE_CONTENT_CONFIG = {
    "model_name": "bert-base-uncased",          # shared transformer
    "length_max": 512,                          # max tokenization length
    "batch_size": 32,                           # batch size for training and evaluation
    "epochs": 5,                                # number of training epochs
    "learning_rate": 2e-5,                      # optimizer learning rate
    "weight_decay": 0.01,                       # L2 regularization
    "dropout_rate": 0.3,                        # dropout used in hybrid model
    "fp16": True,                               # mixed precision training (Transformer model only)
    "output_dir": "models"                      # base directory for outputs
}

HEADLINE_CONFIG = {

}