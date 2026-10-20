# Original CNN weights

`model1.keras` and `model2.keras` are the engineer's supplied Coma/Awake models.
Their bytes are unchanged by the repository cleanup. Loading and preprocessing
live in `flyscope/models/keras.py`; models load lazily with Keras's Torch backend.
They do not have a trained Empty class. See `docs/WORKFLOW.md` for baseline commands.
Newly trained artifacts belong outside this repository.
