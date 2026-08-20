"""
models package
--------------
All ML/DL model definitions, training pipelines, and inference wrappers.

ARCHITECTURE:
  The Digital Twin depends on PredictionInterface, not CNN-BiLSTM directly.
  See prediction_interface.py for the model-agnostic I/O contract.

Model progression (per project roadmap -- student team research decisions in Phase 5):
  Phase 5: Baseline models (statistical, LR, RF, simple LSTM)
  Phase 5: CNN-BiLSTM main model (architecture finalised by student team)
  Phase 6: ML integration via PredictionInterface
  Phase 8: Uncertainty quantification (method TBD)

PLACEHOLDER NOTE:
  Any model that is not trained on real clinical data must set
  PredictionOutput.is_placeholder = True.
"""
