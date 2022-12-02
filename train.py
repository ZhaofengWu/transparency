from paoding.train import train

from data.text_pair_dataset import TextPairDataset
from models.transformer_probe import TransformerProbe

if __name__ == "__main__":
    train(TransformerProbe, TextPairDataset)
