from glycan_profiling.tandem.glycopeptide.scoring.base import GlycopeptideSpectrumMatcherBase
from glycan_profiling.tandem.spectrum_match import Unmodified


class ModelBindingScorer(GlycopeptideSpectrumMatcherBase):
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
        return inst

    def __reduce__(self):
        return self.__class__, (self.tp, self.args, self.kwargs)

    @property
    def model_fit(self):
        try:
            return self.kwargs['model_fits'][0]
        except KeyError:
            try:
                return self.kwargs['model_fit']
            except KeyError:
                return None

    @property
    def model_fits(self):
        try:
            return self.kwargs['model_fits']
        except KeyError:
            return None

    @property
    def partition_label(self):
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

    def __lt__(self, other):
        return self.partition_label < other.partition_label


class DummyScorer(GlycopeptideSpectrumMatcherBase):
    def __init__(self, *args, **kwargs):
        raise TypeError("DummyScorer should not be instantiated!")
