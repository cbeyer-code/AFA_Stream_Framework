# In afa_scores/__init__.py

# Use relative imports to bring the CLASSES into the package's namespace
from .AEDScorer import AEDScorer
from .FeatureScorer import FeatureScorer
from .RandomScorer import RandomScorer

# This is good practice and tells Python what '*' imports should grab
__all__ = ['AEDScorer', 'FeatureScorer', 'RandomScorer']