"""Suppress noisy third-party deprecation warnings that flood training logs.

Import this module BEFORE importing torch / transformers / insightface so the
filters get a chance to register before the underlying modules emit warnings.

Currently silenced:
    * ``torch.cuda.amp.autocast(args...)`` deprecation from wan/* modules.
    * ``torch.load`` ``weights_only=False`` future-default warning.
    * ``numpy.linalg.lstsq`` ``rcond`` parameter warning from insightface.
    * ``pkg_resources is deprecated`` warning from librosa.
    * torchvision ``pretrained=`` / ``weights=`` deprecation from rvm_model.
    * pyloudnorm ``Possible clipped samples in output`` from audio normalize.
    * albumentations ``Error fetching version info`` (offline check_version).
"""
import warnings

_FILTERS = [
    (r"`torch\.cuda\.amp\.autocast.*", FutureWarning),
    (r"`torch\.cpu\.amp\.autocast.*", FutureWarning),
    (r"You are using `torch\.load` with `weights_only=False`.*", FutureWarning),
    (r".*`rcond` parameter will change.*", FutureWarning),
    (r"FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated.*", FutureWarning),
    (r"pkg_resources is deprecated as an API.*", UserWarning),
    (r"The parameter 'pretrained' is deprecated since 0\.13.*", UserWarning),
    (r"Arguments other than a weight enum or `None` for 'weights' are deprecated.*", UserWarning),
    (r"Possible clipped samples in output\.", UserWarning),
    (r"Error fetching version info.*", UserWarning),
]

for pattern, category in _FILTERS:
    warnings.filterwarnings("ignore", message=pattern, category=category)
