GENERAL_CONFIG = {
    "seed": 42,
    "separator": "-" * 40
}

HEADLINE_CONTENT_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",          # shared transformer
    "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_type": ["standard", "hybrid"],
    "model_path_default": ["models/standard/best_model", "models/hybrid/best_model"],
    "length_max": 512,                          # max tokenization length
    "batch_size": 32,                           # batch size for training and evaluation
    "epochs": 3,                                # number of training epochs
    "learning_rate": 2e-5,                      # optimizer learning rate
    "weight_decay": 0.01,                       # L2 regularization
    "dropout_rate": 0.3,                        # dropout used in hybrid model
    "fp16": True,                               # mixed precision training (Transformer model only)
    "output_directory": "models"                # base directory for outputs
}

HEADLINE_CONFIG = {
    "model_type": "logistic",
    "model_path": "models/headline_models/logistic.joblib",

}

DATASETS_CONFIG = {
    "dataset_headline_content_name": "clickbait17",
    "dataset_headline_name": "headlines",
    "dataset_headline2_name": "clickbait_notclickbait_dataset",
    "train_suffix": "train",
    "validation_suffix": "validation",
    "test_suffix": "test",
    "features_suffix": "features"
}

HEADLINE_CONTENT_MODELS_PRETRAINED = [
    "bert-base-uncased",
    "google/bert_uncased_L-2_H-128_A-2",
    "roberta-base",
    "microsoft/deberta-v3-small", # dobre
    "sentence-transformers/all-MiniLM-L6-v2", # takie sobie ale szybkie
    # "khalidalt/DeBERTa-v3-large-mnli", # dużo VRAMu, trzeba batch size do 8 zmniejszyć i czekać 6 godzin
    "distilroberta-base",
    "allenai/longformer-base-4096",
    # TODO add sentence transformer
    # ALBERT
    # Cross encoders
    "cross-encoder/ms-marco-MiniLM-L6-v2" # cross encoder, gorszy niż all-MiniLM (podobny czas)
]

    # ['AlbertTokenizer', 'BartTokenizer', 'BarthezTokenizer', 'BertTokenizer', 'BigBirdTokenizer', 'BlenderbotTokenizer', 'CamembertTokenizer', 'CLIPTokenizer', 'CodeGenTokenizer', 'ConvBertTokenizer', 'DebertaTokenizer', 'DebertaV2Tokenizer', 'DistilBertTokenizer', 'DPRReaderTokenizer', 'DPRQuestionEncoderTokenizer', 'DPRContextEncoderTokenizer', 'ElectraTokenizer', 'FNetTokenizer', 'FunnelTokenizer', 'GPT2Tokenizer', 'HerbertTokenizer', 'LayoutLMTokenizer', 'LayoutLMv2Tokenizer', 'LayoutLMv3Tokenizer', 'LayoutXLMTokenizer', 'LongformerTokenizer', 'LEDTokenizer', 'LxmertTokenizer', 'MarkupLMTokenizer', 'MBartTokenizer', 'MBart50Tokenizer', 'MPNetTokenizer', 'MobileBertTokenizer', 'MvpTokenizer', 'NllbTokenizer', 'OpenAIGPTTokenizer', 'PegasusTokenizer', 'Qwen2Tokenizer', 'RealmTokenizer', 'ReformerTokenizer', 'RemBertTokenizer', 'RetriBertTokenizer', 'RobertaTokenizer', 'RoFormerTokenizer', 'SeamlessM4TTokenizer', 'SqueezeBertTokenizer', 'T5Tokenizer', 'UdopTokenizer', 'WhisperTokenizer', 'XLMRobertaTokenizer', 'XLNetTokenizer', 'SplinterTokenizer', 'XGLMTokenizer', 'LlamaTokenizer', 'CodeLlamaTokenizer', 'GemmaTokenizer', 'Phi3Tokenizer']

ARTICLE_SCRAPING_CONFIG = {
    "content_length_min": 50,

}