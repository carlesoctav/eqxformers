from transformers import BertConfig
import time
from src.eqxformers.models.bert import BertForMaskedLM 
import equinox as eqx
import jax 


def main():
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        _attn_implementation="eager",
    )

    config2b = BertConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=10240,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        _attn_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_scan = False
    )


    @eqx.filter_jit
    def init_module(key):
        return BertForMaskedLM(config2b, key = key)



    start = time.monotonic()
    lower = init_module.lower(jax.random.key(10))
    hlo_txt = lower.as_text() 
    compile = lower.compile()
    with open("./init_module_new_2_5b_no_scan.txt", "w") as f:
        f.write(hlo_txt) 

    mod= compile(jax.random.key(10))
    jax.block_until_ready(jax.tree.leaves(mod))
    end = time.monotonic() - start
    print(f"DEBUGPRINT[89]: init_time.py:28: end={end}")



if __name__ == "__main__":
    main()
