
import torch
import torch.nn as nn
import timm
import torchvision
import torch.nn.functional as F

import  numpy as np

# model adapted from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/data



class Encoder(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)


        self.expand=nn.Linear(in_features=64,out_features=193)

    def forward(self, x):
        bs = x.size(0)
        x=x/255.
        features = self.cnn.forward_features(x)

        features = features.view(bs,features.size(1),-1)

        features = self.expand(features)

        features = features.permute([0,2,1])
        return features
class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, ):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(Decoder, self).__init__()


        self.rnn=nn.LSTM(input_size=1280,
                         hidden_size=512,
                         dropout=0.3,
                         bidirectional=True,
                         num_layers=2)


        self.head=nn.Linear(1024,274)
        self.init_weights()

    def init_weights(self):

        self.head.bias.data.fill_(0)
        self.head.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune


    def forward(self, cnn_fatures, encoded_captions):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        """

        out, c = self.rnn(cnn_fatures)
        out = self.head(out)
        out = out.permute([0, 2, 1])

        return out






class Caption(nn.Module):
    def __init__(self,
                 embed_dim,
                 attention_dim,
                 decoder_dim,
                 encoder_dim,
                 vocab_size,
                 dropout,
                 max_length,
                 tokenizer,
                 ):
        super().__init__()
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.token=tokenizer

        self.max_length=max_length
    def forward(self, images,labels=None):



        features = self.encoder(images)
        predictions, alphas = self.decoder(features, labels)

        return predictions





if __name__=='__main__':
    dummy_input = torch.randn(2, 3, 256, 256, device='cpu')
    embed_dim = 200
    vocab_size = 193  ##len(vocab)
    attention_dim = 300
    encoder_dim = 512
    decoder_dim = 300
    import sys
    sys.path.append('.')
    from make_data import Tokenizer

    def get_token():
        token_tools = Tokenizer()
        token_tools.stoi = np.load("../tokenizer.stio.npy", allow_pickle=True).item()
        token_tools.itos = np.load("../tokenizer.itos.npy", allow_pickle=True).item()

        return token_tools


    token_tools = get_token()
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = Caption(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        dropout=0.5,
        tokenizer=token_tools,
        max_length=275

    ).to(device)

    ### load your weights
    model.eval()

    import sys

    predictions = model(dummy_input)

    predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
    print(predicted_sequence.shape)
    _text_preds = token_tools.predict_captions(predicted_sequence)
    print(_text_preds)


