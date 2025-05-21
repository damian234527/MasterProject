GENERAL_CONFIG = {
    "seed": 42

}

HEADLINE_CONTENT_CONFIG = {
    "model_name": "bert-base-uncased",          # shared transformer
    "tokenizer_name": "bert-base-uncased",
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
    "roberta-base",
    "microsoft/deberta-v3-small" # dobre
    "sentence-transformers/all-MiniLM-L6-v2", # takie sobie ale szybkie
    "khalidalt/DeBERTa-v3-large-mnli", # dużo VRAMu, trzeba batch size do 8 zmniejszyć i czekać 6 godzin
    "Longformer-base-4096",
    # TODO add sentence transformer
]

    # ['AlbertTokenizer', 'BartTokenizer', 'BarthezTokenizer', 'BertTokenizer', 'BigBirdTokenizer', 'BlenderbotTokenizer', 'CamembertTokenizer', 'CLIPTokenizer', 'CodeGenTokenizer', 'ConvBertTokenizer', 'DebertaTokenizer', 'DebertaV2Tokenizer', 'DistilBertTokenizer', 'DPRReaderTokenizer', 'DPRQuestionEncoderTokenizer', 'DPRContextEncoderTokenizer', 'ElectraTokenizer', 'FNetTokenizer', 'FunnelTokenizer', 'GPT2Tokenizer', 'HerbertTokenizer', 'LayoutLMTokenizer', 'LayoutLMv2Tokenizer', 'LayoutLMv3Tokenizer', 'LayoutXLMTokenizer', 'LongformerTokenizer', 'LEDTokenizer', 'LxmertTokenizer', 'MarkupLMTokenizer', 'MBartTokenizer', 'MBart50Tokenizer', 'MPNetTokenizer', 'MobileBertTokenizer', 'MvpTokenizer', 'NllbTokenizer', 'OpenAIGPTTokenizer', 'PegasusTokenizer', 'Qwen2Tokenizer', 'RealmTokenizer', 'ReformerTokenizer', 'RemBertTokenizer', 'RetriBertTokenizer', 'RobertaTokenizer', 'RoFormerTokenizer', 'SeamlessM4TTokenizer', 'SqueezeBertTokenizer', 'T5Tokenizer', 'UdopTokenizer', 'WhisperTokenizer', 'XLMRobertaTokenizer', 'XLNetTokenizer', 'SplinterTokenizer', 'XGLMTokenizer', 'LlamaTokenizer', 'CodeLlamaTokenizer', 'GemmaTokenizer', 'Phi3Tokenizer']
