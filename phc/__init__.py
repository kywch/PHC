from pathlib import Path
from types import SimpleNamespace

# Absolute path for loading resources
PHC_ROOT = Path('/'.join(__path__[0].split('/')[:-1]))

BODY_MODEL_DIR = PHC_ROOT / 'phc/data/smpl'

# TODO: Remove this. This is used globally
flags = SimpleNamespace(
    test=False,
    debug=False,
    real_traj=False,
    im_eval=False,
)
