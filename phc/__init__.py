from pathlib import Path

# Absolute path for loading resources
PHC_ROOT = Path('/'.join(__path__[0].split('/')[:-1]))

BODY_MODEL_DIR = PHC_ROOT / 'phc/data/smpl'
