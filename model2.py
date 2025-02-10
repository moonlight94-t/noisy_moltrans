import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)

class BIN_Interaction_Flat1(nn.Sequential):   
    def __init__(self, **config):
        super().__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate'][0]
        
        #densenet
        self.kernal_dense_size = config['kernal_dense_size'] #?
        self.batch_size = config['batch_size'][0]
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        
        #encoder
        self.n_layer = 2 # encoder layer 개수 
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob'][0]
        self.hidden_dropout_prob = config['hidden_dropout_prob'][0]
        
        self.flatten_dim = config['flat_dim'] 
        
        # specialized embedding with positional one
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate) # input_dim_drug랑 max_d랑 어떻게 다른거지 -> FCS vocab 크기, 각 drug의 maximum sequence length 인가 
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        
        self.icnn = nn.Conv2d(1, 3, 3, padding = 0) # in_channel size, out_channel size, kernel size
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )
        
    def forward(self, d, p, d_mask, p_mask):
        
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2) # (batch_size, hidden_size) -> (batch_size, 1, 1, hidden_size)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0 # 무시할 위치을 1로 만들어 준 후 큰 음수를 곱해 softmax 결과에서 0에 가까운 값 나오게 만듬 
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        
        d_emb = self.demb(d) # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float()) # (batch_size, seq_length, hidden_size)
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())

        # repeat to have the same tensor size for aggregation   
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1) # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1) # repeat along drug size
        
        i = d_aug * p_aug # interaction, element-wise 연산 
        i_v = i.view(int(self.batch_size), -1, self.max_d, self.max_p) # 데이터의 배치는 바꾸지 않고 데이터를 바라보는 관점을 바꾸는 방법(conv 적용할 때 채널위치로 둘려고) , self.gpus로 나눠주는 이유를 모르겠음?
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim = 1)
        i_v = torch.unsqueeze(i_v, 1)
        
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        
        #f = self.icnn2(self.icnn1(i_v))
        f = self.icnn(i_v)
        
        f = f.view(int(self.batch_size), -1) #flatten 역할 
        
        #f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)
        
        #score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score
    
class BIN_Interaction_Flat2(nn.Sequential):   
    def __init__(self, **config):
        super().__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate'][1]
        
        #densenet
        self.kernal_dense_size = config['kernal_dense_size'] #?
        self.batch_size = config['batch_size'][1]
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        
        #encoder
        self.n_layer = 3 # encoder layer 개수 
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob'][1]
        self.hidden_dropout_prob = config['hidden_dropout_prob'][1]
        
        self.flatten_dim = config['flat_dim'] 
        
        # specialized embedding with positional one
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate) # input_dim_drug랑 max_d랑 어떻게 다른거지 -> FCS vocab 크기, 각 drug의 maximum sequence length 인가 
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        
        self.icnn = nn.Conv2d(3, 3, 3, padding = 0) # in_channel size, out_channel size, kernel size
        self.icnn2 = nn.Conv2d(1,3,3,padding=1)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1))
            
    def forward(self, d, p, d_mask, p_mask):
        
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2) # (batch_size, hidden_size) -> (batch_size, 1, 1, hidden_size)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0 # 무시할 위치을 1로 만들어 준 후 큰 음수를 곱해 softmax 결과에서 0에 가까운 값 나오게 만듬 
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        
        d_emb = self.demb(d) # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float()) # (batch_size, seq_length, hidden_size)
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())

        # repeat to have the same tensor size for aggregation   
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1) # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1) # repeat along drug size
        
        i = d_aug * p_aug # interaction, element-wise 연산 
        i_v = i.view(int(self.batch_size), -1, self.max_d, self.max_p) # 데이터의 배치는 바꾸지 않고 데이터를 바라보는 관점을 바꾸는 방법(conv 적용할 때 채널위치로 둘려고) , self.gpus로 나눠주는 이유를 모르겠음?
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim = 1)
        i_v = torch.unsqueeze(i_v, 1)
        
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        
        f = self.icnn(self.icnn2(i_v))
        
        f = f.view(int(self.batch_size), -1) #flatten 역할 
        
        #f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)
        
        #score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score
    
class BIN_Interaction_Flat3(nn.Sequential):   
    def __init__(self, **config):
        super().__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate'][2]
        
        #densenet
        self.kernal_dense_size = config['kernal_dense_size'] #?
        self.batch_size = config['batch_size'][2]
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        
        #encoder
        self.n_layer = 3 # encoder layer 개수 
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob'][2]
        self.hidden_dropout_prob = config['hidden_dropout_prob'][2]
        
        self.flatten_dim = config['flat_dim'] 
        
        # specialized embedding with positional one
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate) # input_dim_drug랑 max_d랑 어떻게 다른거지 -> FCS vocab 크기, 각 drug의 maximum sequence length 인가 
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        
        self.icnn2= nn.Conv2d(1, 5, 3, padding = 1)
        self.icnn3=nn.Conv2d(5,10,3,padding=1)
        self.icnn = nn.Conv2d(10, 3, 3, padding = 0) # in_channel size, out_channel size, kernel size
        
        self.icnn = nn.Conv2d(3, 3, 3, padding = 0) # in_channel size, out_channel size, kernel size
        self.icnn2 = nn.Conv2d(1,3,3,padding=1)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1))
        
    def forward(self, d, p, d_mask, p_mask):
        
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2) # (batch_size, hidden_size) -> (batch_size, 1, 1, hidden_size)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0 # 무시할 위치을 1로 만들어 준 후 큰 음수를 곱해 softmax 결과에서 0에 가까운 값 나오게 만듬 
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        
        d_emb = self.demb(d) # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float()) # (batch_size, seq_length, hidden_size)
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())

        # repeat to have the same tensor size for aggregation   
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1) # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1) # repeat along drug size
        
        i = d_aug * p_aug # interaction, element-wise 연산 
        i_v = i.view(int(self.batch_size), -1, self.max_d, self.max_p) # 데이터의 배치는 바꾸지 않고 데이터를 바라보는 관점을 바꾸는 방법(conv 적용할 때 채널위치로 둘려고) , self.gpus로 나눠주는 이유를 모르겠음?
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim = 1)
        i_v = torch.unsqueeze(i_v, 1)
        
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        
        f = self.icnn(self.icnn2(i_v))
        
        f = f.view(int(self.batch_size), -1) #flatten 역할 
        
        #f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)
        
        #score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score
   
# help classes    
    
class LayerNorm(nn.Module): # hidden_size, 왜 따로 정의했지? -> gamma, beta 값 학습시키려고
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #batch_size로 확장 
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings) # embedding vector의 요소가 무작위로 비활성화 0 
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size) # hidden_size=all_head_size, input features 간의 상관관계 학습 
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x): # x : (batch_size, seq_length, hidden_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # batch_size, num_attention_heads, seq_length, attention_head_size / seq_length : 토큰의 총 개수, hidden_size : 각 토큰이 임베딩 된 차원의 크기

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (batch_size, num_heads, seq_length, head_size) * (batch_size,num_heads,head_size,seq_length)=(batch_size,num_heads,seq_length,_seq_length)
        # matmul 뒤의 두 차원에서 행렬곱 연산, AdotB=A*transposeB
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 분산이 sqrt(size)에 비례해서 커지므로 정규화해주는 역할 

        attention_scores = attention_scores + attention_mask # attention_mask 입력 시퀀스의 길이가 다를 경우 패딩이 들어가는데 attention연산에서는 무의미하므로 비활성화 해주는 역할 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # (batch, num_heads, seq_length, seq_length) * (batch, num_heads, seq_length, head_size)=(batch, num_heads, seq_length, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (batch, seg_length, num_heads, head_size) , .contiguous() 메모리에 연속적인 형태로 복사하여 저장 -> 이후 연산과정에서 메모리 접근이 빠르고 효율적 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) #(batch, seg_length, all_head_size)
        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size) # attention 결과를 선형 변환, 결과를 한 번 더 조정해서 중요한 피처를 강화하거나 불필요한 정보를 줄임 
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # residual connection
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

    
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states