

class Config():
    def __init__(self):

        self.architectures= ["BertForMaskedLM" ]
        self.attention_probs_dropout_prob= 0.1
        self.directionality= "bidi"
        self.finetuning_task= None
        self.hidden_act= "gelu_new"
        self.hidden_dropout_prob= 0.1
        self.hidden_size= 768
        self.initializer_range= 0.02
        self.intermediate_size= 3072
        self.layer_norm_eps= 1e-12
        self.loss_type= "ce"
        self.max_position_embeddings= 128
        self.model_type= "bert"
        self.num_attention_heads= 12
        self.num_hidden_layers= 12
        self.num_labels= 34
        self.output_attentions= False
        self.output_hidden_states= False
        self.output_past= True
        self.pad_token_id= 0
        self.pooler_fc_size= 768
        self.pooler_num_attention_heads= 12
        self.pooler_num_fc_layers= 3
        self.pooler_size_per_head= 128
        self.pooler_type= "first_token_transform"
        self.pruned_heads= {}
        self.torchscript= False
        self.type_vocab_size= 2
        self.use_bfloat16= False
        self.vocab_size= 21128