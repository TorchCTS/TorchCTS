# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def get_gpt2_config():
    try:
        from transformers import GPT2Config
        return GPT2Config(
            n_positions=128,
            n_ctx=128,
            n_embd=64,
            n_layer=2,
            n_head=4,
            vocab_size=1000,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
    except ImportError:
        return None

def get_bert_config():
    try:
        from transformers import BertConfig
        return BertConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            max_position_embeddings=128,
            vocab_size=1000,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    except ImportError:
        return None

def get_qwen2_config():
    try:
        from transformers import Qwen2Config
        return Qwen2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2, # GQA
            intermediate_size=128,
            num_hidden_layers=2,
            max_position_embeddings=128,
            sliding_window=64,
            vocab_size=1000,
            attention_dropout=0.0,
        )
    except ImportError:
        return None
