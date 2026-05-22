Pipeline
--------
- Data pipeline scripts for simulation, embedding, indexing, and preparing training data.
- Typical flow:
  1. `simulate_users.py`
  2. `embed_items.py`
  3. `build_index.py`
  4. `prepare_training_data.py`
  5. `train_two_tower.py` / `train_ranker.py`
- Run these locally before serving recommendations.
