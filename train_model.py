
# %load_ext tensorboard

# %%capture
# from google.colab import drive
# drive.mount('/content/drive')

!pip install datasets
!pip install tokenizers
!pip install torchmetrics

!git clone https://github.com/omArray99/MT_en-es_transformer.git

# %%capture
# %cd /content/MT_en-es_transformer

# %tensorboard --logdir runs

!mkdir -p /content/drive/MyDrive/Models/MT_en-es_transformer/weights
!mkdir -p /content/drive/MyDrive/Models/MT_en-es_transformer/vocab

from config import fetch_configuration
cfg = fetch_configuration()
cfg['model_dir'] = '..//drive/MyDrive/Models/MT_en-es_transformer/weights'
cfg['tokenizer_path'] = '..//drive/MyDrive/Models/MT_en-es_transformer/vocab/tokenizer_{0}.json'
cfg['batch_size'] = 24
cfg['num_epochs'] = 20
cfg['preload'] = None

from train import train_model

train_model(cfg)

