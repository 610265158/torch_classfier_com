import math
import torch
import torch.nn as nn
import timm
import torchvision
import torch.nn.functional as F
import torchvision.models
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Encoder(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)

        self.reduce_head = nn.Sequential(nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                         nn.BatchNorm2d(512, ),
                                         nn.ReLU(inplace=True))

    def forward(self, x):
        bs = x.size(0)
        x = x / 255.
        x = torch.cat([x,x,x],dim=1)
        features = self.cnn.forward_features(x)
        features = self.reduce_head(features)
        features = features.view(bs, features.size(1),-1,)

        features = features.permute(2,0,1)

        return features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # self.len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):


        x = x + self.pe[:x.size(1), :]

        return self.dropout(x)


class TransformerDecoderLayer_pp(nn.TransformerDecoderLayer):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class Decoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 n_head=8,
                 num_layer=1,
                 vocable_size=193,
                 dropout=0.1,
                 length=275):
        super(Decoder, self).__init__()
        self.decoder_layer  = TransformerDecoderLayer_pp(d_model=d_model, nhead=n_head)
        self.transformdecoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layer)

        self.fc_out = nn.Linear(d_model, vocable_size)

        self.embedding = nn.Embedding(vocable_size, d_model)

        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len=length)



        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.vocable_size=vocable_size
    def forward(self, memory, tgc=None, tgt_mask=None):
        tgc = self.embedding(tgc)

        tgc = self.pos_embedding(tgc)
        tgc=tgc.permute(1,0,2)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgc)).to(self.device)

        if memory.size()[-1] == tgc.size()[-1]:
            output = self.transformdecoder(tgc, memory, tgt_mask=tgt_mask)
            output = self.fc_out(output)
        else:
            print("src and tgc dim different!")


        ##ouput shape SNE,  permute to NSE
        output=output.permute(1,0,2)
        return output[:,1:,:]

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def predict(self, memory, decode_length,tokenizer):
        """
        the memory is 85 X 1 x d_model,output is list
        """
        bs = memory.size()[1]
        pred_num = torch.full((1, bs), tokenizer.stoi["<sos>"]).to(self.device)
        # pred = self.decoder(pred_num)
        pred_list = torch.zeros((bs, decode_length), dtype=torch.long).to(self.device)
        pred_list[:, 0] = pred_num.reshape(-1)
        predictions=torch.zeros((bs, decode_length-1,self.vocable_size)).to(self.device)


        for i in range(1, decode_length):

            pred = self.embedding(pred_list[:, :i])
            pred = self.pos_embedding(pred)

            pred = pred.permute(1, 0, 2)

            # print(pred.shape)
            output = self.transformdecoder(pred, memory)
            output = self.fc_out(output)

            cur_output=output[i-1,...]

            predictions[:,i-1,:]=cur_output

            output = cur_output.argmax(1).reshape(-1)
            # print(output.shape)
            pred_list[:, i] = output
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


        self.token = tokenizer

        self.max_length = max_length+1 ##because it must from start signal differ from lstm

        self.encoder = Encoder()
        self.decoder = Decoder(length=self.max_length)


    def forward(self, images, labels=None, label_length=None):

        if labels is not None:

            features = self.encoder(images)
            predictions = self.decoder(features, labels)

            return predictions
        else:
            features = self.encoder(images)
            predictions = self.decoder.predict(features, self.max_length, self.token)

            return predictions


if __name__ == '__main__':
    dummy_input = torch.randn(2, 1, 256, 256, device='cuda')
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
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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

    predicted_sequence = model(dummy_input).cpu().numpy()

    print(predicted_sequence.shape)
    _text_preds = token_tools.predict_captions(predicted_sequence)
    print(_text_preds)


