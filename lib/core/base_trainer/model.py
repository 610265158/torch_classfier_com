
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

        self.reduce_head=nn.Sequential(nn.Conv2d(1280,512,kernel_size=1,stride=1,padding=0),
                                       nn.BatchNorm2d(512,),
                                       nn.ReLU())
    def forward(self, x):
        bs = x.size(0)
        x=x/255.
        features = self.cnn.forward_features(x)
        features = self.reduce_head(features)
        features = features.permute(0, 2, 3, 1)
        features = features.view(bs,-1,features.size(-1))
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


class DecoderWithAttention(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, cnn_fatures, encoded_captions,max_length):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer

        """
        batch_size = cnn_fatures.size(0)
        num_pixels = cnn_fatures.size(1)
        vocab_size = self.vocab_size



        # embedding transformed sequence for vector
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(cnn_fatures)  # (batch_size, decoder_dim)
        # set decode length by caption length - 1 because of omitting start token

        predictions = torch.zeros(batch_size,max_length, vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(self.device)
        # predict sequence
        for t in range(max_length):

            attention_weighted_encoding, alpha = self.attention(cnn_fatures, h)
            gate = self.sigmoid(self.f_beta(h[:]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1),
                (h[:], c[:]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        return predictions, encoded_captions

    def predict(self, cnn_fatures, decode_lengths, tokenizer):
        batch_size = cnn_fatures.size(0)

        vocab_size = self.vocab_size


        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(self.device) * tokenizer.stoi["<sos>"]
        embeddings = self.embedding(start_tockens)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(cnn_fatures)  # (batch_size, decoder_dim)
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)
        # predict sequence
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(cnn_fatures, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == tokenizer.stoi["<eos>"]:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions


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
        self.decoder=DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=encoder_dim, dropout=dropout)
        self.token=tokenizer

        self.max_length=max_length
    def forward(self, images,labels=None,train_length=None):

        if labels is not None:

            features = self.encoder(images)
            predictions, alphas = self.decoder(features, labels,train_length)

            return predictions,alphas
        else:
            features = self.encoder(images)
            predictions = self.decoder.predict(features, self.max_length, self.token)

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

    predictions = model.predict(dummy_input)
    predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
    print(predicted_sequence.shape)
    _text_preds = token_tools.predict_captions(predicted_sequence)
    print(_text_preds)


