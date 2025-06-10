import json
class Transformer_hyper:
    def __init__(self, num_layers=6, d_model=512, num_heads=8, num_kv_heads = None, ffn_dim = 2048):
        self.name = "default_transformer"
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_dim = ffn_dim

    def __repr__(self):
        return (f"{self.name}(num_layers={self.num_layers}, "
                f"d_model={self.d_model}, num_heads={self.num_heads}, "
                f"dff={self.dff}, dropout_rate={self.dropout_rate})")
    
    def read_from_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.name = data.get('name', 'Transformer_hyper')
            self.num_layers = data.get('num_layers', self.num_layers)
            self.d_model = data.get('model_dimension', self.d_model)
            self.num_heads = data.get('num_attention_heads', self.num_heads)
            self.ffn_dim = data.get('ffn_dimension', self.ffn_dim)
            self.num_kv_heads = data.get('num_key_value_heads', self.num_kv_heads)