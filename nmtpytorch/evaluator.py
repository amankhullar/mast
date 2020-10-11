# -*- coding: utf-8 -*-
from collections import OrderedDict

from . import metrics
from .utils.filterchain import FilterChain
from .utils.misc import get_language
from rouge import Rouge as newRouge

import os

class Evaluator:
    def __init__(self, refs, beam_metrics, filters=''):
        # metrics: list of upper-case beam-search metrics
        self.kwargs = {}
        self.scorers = OrderedDict()
        self.refs = list(refs.parent.glob(refs.name))
        self.language = get_language(self.refs[0])
        if self.language is None:
            # Fallback to en (this is only relevant for METEOR)
            self.language = 'en'

        self.filter = lambda s: s
        if filters:
            self.filter = FilterChain(filters)
            self.refs = self.filter(refs)

        assert len(self.refs) > 0, "Number of reference files == 0"

        for metric in sorted(beam_metrics):
            self.kwargs[metric] = {'language': self.language}
            self.scorers[metric] = getattr(metrics, metric + 'Scorer')()

    def score(self, hyps):
        """hyps is a list of hypotheses as they come out from decoder."""
        assert isinstance(hyps, list), "hyps should be a list."
        new_rouge = newRouge()

        # Post-process if requested
        hyps = self.filter(hyps)

        with open(self.refs[0]) as f:
            ref_sents = f.read().strip().split('\n')

        #print("hyp is : {}".format(hyps))
        for hyp, ref in zip(hyps, ref_sents):
            ref = ref.split(" ")
            file_name = ref[0]
            ref = " ".join(ref[1:])
            with open(os.path.join('/home/aman_khullar/att_wts/', file_name, 'ref.txt'), 'w') as f:
                f.write(ref)

            hyp = hyp.split(" ")
            hyp = " ".join(hyp[1:])
            with open(os.path.join('/home/aman_khullar/att_wts/', file_name, 'hyp.txt'), 'w') as f:
                f.write(hyp)

        new_hyps = [" ".join(hyp.split(" ")[1:]) for hyp in hyps]
        new_refs = [" ".join(ref.split(" ")[1:]) for ref in ref_sents]
        new_rouge_scores = new_rouge.get_scores(new_hyps, new_refs, avg=True)
        print("Rouge 1 score is : {0:.3f}".format(new_rouge_scores['rouge-1']['f']*100))
        print("Rouge 2 score is : {0:.3f}".format(new_rouge_scores['rouge-2']['f']*100))
        print(new_rouge_scores)
        results = []
        for key, scorer in self.scorers.items():
            results.append(
                scorer.compute(self.refs, hyps, **self.kwargs[key]))
        return results
        
