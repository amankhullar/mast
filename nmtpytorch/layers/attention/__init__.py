from .mlp import MLPAttention
from .dot import DotAttention
from .hierarchical import HierarchicalAttention
from .hierarchical_coverage import HierarchicalAttentionCoverage
from .bidaf_att import BiDAFAttention
from .co import CoAttention
from .mhco import MultiHeadCoAttention
from .uniform import UniformAttention


def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'dot': DotAttention,
        'hier': HierarchicalAttention,
        'hiercov': HierarchicalAttentionCoverage,
        'co': CoAttention,
        'mhco': MultiHeadCoAttention,
        'uniform': UniformAttention,
        'bidaf': BiDAFAttention,
    }[type_]
