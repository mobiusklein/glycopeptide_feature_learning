from typing import Any, Dict, Iterator, List, Optional, Type

from glycan_profiling.tandem.glycopeptide.scoring.base import GlycopeptideSpectrumMatcherBase
from glycan_profiling.tandem.spectrum_match import Unmodified

from glycopeptide_feature_learning.multinomial_regression import MultinomialRegressionFit
from glycopeptide_feature_learning.partitions import SplitModelFit, partition_cell_spec

from ._c.score_set import ModelScoreSet
from .fdr_model import CorrelationPeptideSVMModel

class ModelBindingScorer(GlycopeptideSpectrumMatcherBase):
    partition_key: int
    tp: Type[GlycopeptideSpectrumMatcherBase]
    args: tuple
    kwargs: Dict[str, Any]

    def __init__(self, tp, args=None, kwargs=None, *_args, **_kwargs):
        if args is None:
            args = tuple(_args)
        else:
            args = tuple(args) + tuple(_args)
        if kwargs is None:
            kwargs = _kwargs
        else:
            kwargs.update(_kwargs)
        self.tp = tp
        self.args = args
        self.kwargs = kwargs
        self.partition_key = -1

    def __repr__(self):
        return "ModelBindingScorer(%s)" % (repr(self.tp),)

    def __eq__(self, other):
        try:
            return (self.tp == other.tp) and (self.args == other.args) and (self.kwargs == other.kwargs)
        except AttributeError:
            return False

    def __call__(self, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        kwargs.update(self.kwargs)
        args = self.args + args
        return self.tp(scan, target, mass_shift=mass_shift, *args, **kwargs)

    def evaluate(self, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        inst = self.tp(scan, target, mass_shift=mass_shift, *self.args, **self.kwargs)
        inst.match(*args, **kwargs)
        inst.calculate_score(*args, **kwargs)
        inst.partition_key = self.partition_key
        return inst

    def __reduce__(self):
        return self.__class__, (self.tp, self.args, self.kwargs)

    @property
    def model_fit(self) -> Optional[MultinomialRegressionFit]:
        try:
            return self.kwargs['model_fits'][0]
        except KeyError:
            try:
                return self.kwargs['model_fit']
            except KeyError:
                return None

    @property
    def model_fits(self) -> Optional[List[MultinomialRegressionFit]]:
        try:
            return self.kwargs['model_fits']
        except KeyError:
            return None

    @property
    def model_selectors(self) -> Optional[SplitModelFit]:
        try:
            return self.kwargs['model_selectors']
        except KeyError:
            return None

    def itermodels(self) -> Iterator[MultinomialRegressionFit]:
        if self.model_fits:
            yield from iter(self.model_fits)
        elif self.model_selectors:
            yield from self.model_selectors
        else:
            yield self.model_fit

    @property
    def partition_label(self) -> Optional[partition_cell_spec]:
        try:
            return self.kwargs['partition']
        except KeyError:
            return None

    def compact(self):
        model_fits = self.model_fits
        if model_fits is None:
            model_fit = self.model_fit
            if model_fit is not None:
                model_fit.compact()
        else:
            for model_fit in model_fits:
                model_fit.compact()
            last = model_fits[0]
            for model_fit in model_fits[1:]:
                if model_fit.reliability_model == last.reliability_model:
                    model_fit.reliability_model = last.reliability_model
        model_selectors = self.model_selectors
        if model_selectors is not None:
            for model_fit in model_selectors:
                model_fit.compact()

    def __lt__(self, other):
        return self.partition_label < other.partition_label


class DummyScorer(GlycopeptideSpectrumMatcherBase):
    def __init__(self, *args, **kwargs):
        raise TypeError("DummyScorer should not be instantiated!")


class HelperMethods:

    @classmethod
    def get_score_set_type(cls):
        return ModelScoreSet

    @classmethod
    def get_fdr_model_for_dimension(cls, label: str):
        if label == 'peptide':
            return CorrelationPeptideSVMModel
