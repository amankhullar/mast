# -*- coding: utf-8 -*-
import torch
from torch import nn

from ...utils.nn import get_activation_fn

class HierarchicalAttentionCoverage(nn.Module):
    """Hierarchical attention over multiple modalities with coverage."""
    def __init__(self, ctx_dims, hid_dim, mid_dim, att_activ='tanh'):
        super().__init__()

        self.activ = get_activation_fn(att_activ)
        self.ctx_dims = ctx_dims
        self.hid_dim = hid_dim
        self.mid_dim = mid_dim

        self.ctx_projs = nn.ModuleList([
            nn.Linear(dim, mid_dim, bias=False) for dim in self.ctx_dims])
        self.dec_proj = nn.Linear(hid_dim, mid_dim, bias=True)
        self.mlp = nn.Linear(self.mid_dim, 1, bias=False)

        self.coverage_feature = nn.Linear(1, self.mid_dim)

    def forward(self, contexts, hid, coverage):
        # TODO: Handle initial coverage case
        if coverage is not None:
            coverage_feature = self.coverage_feature(coverage)

            dec_state_proj = self.dec_proj(hid)
            ctx_projected = torch.cat([
                p(ctx).unsqueeze(0) for p, ctx
                in zip(self.ctx_projs, contexts)], dim=0)
            #print("dec state size : {} \n ctx_projected size : {} \n cov_feat size : {}".format(dec_state_proj.size(), ctx_projected.size(), coverage_feature.size()))
            energies = self.mlp(self.activ(dec_state_proj + ctx_projected + coverage_feature))
            att_dist = nn.functional.softmax(energies, dim=0)

            coverage = coverage + att_dist
        else:
            dec_state_proj = self.dec_proj(hid)
            ctx_projected = torch.cat([
                p(ctx).unsqueeze(0) for p, ctx
                in zip(self.ctx_projs, contexts)], dim=0)
            energies = self.mlp(self.activ(dec_state_proj + ctx_projected))
            att_dist = nn.functional.softmax(energies, dim=0)

            #print("coverage size : {}".format(att_dist.size()))
            coverage = att_dist

        ctxs_cat = torch.cat([c.unsqueeze(0) for c in contexts])
        joint_context = (att_dist * ctxs_cat).sum(0)

        return att_dist, joint_context, coverage