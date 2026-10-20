# ECOBIO Internship — FlyScope

FlyScope prepares annotated Drosophila experiments and compares coma-recovery
models. The local app has separate **CCRT** and **Egg counting** pages. Egg
counting remains an annotation skeleton until its experimental format is known.

Start with the **[complete workflow](docs/WORKFLOW.md)**: one Python environment,
one root `requirements.txt`, GPU checks, annotations/exports, pose preparation,
matched training and independent evaluation. Run all commands from this repository
root; the old `CCRT/logiciel-v2_2` working directory has been retired.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Use Python 3.12.12 and the OS libraries described in the workflow. Linux Torch
wheels support the RTX PRO Blackwell GPU; CPU execution is also available.

| Location | Purpose |
| --- | --- |
| `flyscope/` | Maintained Python package: imaging, datasets, models, training, evaluation and reporting |
| `frontend/` | Active browser assets and templates only |
| `tools/` | Stable command launchers and runtime/validation utilities |
| `tests/` | Python, JavaScript and browser checks |
| `models/legacy/` | Two supplied original CNN weights, retained unchanged |
| `docs/` | Workflow, app guide, experiment audit and maintenance instructions |
| `archive/original-comparisons/` | Historical sample data; never loaded by the app |

Use the existing command names, such as `python tools/model_suite.py compare`.
Implementations live in the package rather than in executable scripts. Keep
new experiments, exported datasets, caches, trained weights and results outside Git.

- [App guide](docs/APP_GUIDE.md): annotation, manual corrections and exports.
- [First training audit](docs/TRAINING_REVIEW.md): measured problems and fixes.
- [Maintaining the project](docs/MAINTAINING.md): responsibilities, scientific contracts and checks.
