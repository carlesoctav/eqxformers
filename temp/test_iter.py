from src.eqxformers.models.bert import BertModel, BertConfig
import jax
import jax.tree_util as jtu


a = BertModel(BertConfig(use_scan = False), key = jax.random.key(10))


def f(path, leaf):
    print(f"DEBUGPRINT[22]: test_iter.py:10: path={type(path)}")
    print(f"DEBUGPRINT[22]: test_iter.py:10: path={path}")
    path = jtu.keystr(path)
    print(f"DEBUGPRINT[21]: test_iter.py:10: path={type(path)}")
    print(f"DEBUGPRINT[20]: test_iter.py:8: path={path}")



jtu.tree_map_with_path(f, a)
