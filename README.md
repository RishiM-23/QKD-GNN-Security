🌌 **QKD-GNN: Quantum Key Distribution Attack Localization**

A machine learning pipeline using Edge-Centric Graph Neural Networks to detect and pinpoint physical-layer attacks (Intercept-Resend, Blinding) in QKD networks.

🎯 **Overview**

Pinpointing eavesdroppers in multi-hop QKD networks is computationally complex. Traditional network-wide thresholds often fail against stealthy, localized attacks. This project solves this by feeding multi-hop telemetry into a Graph Neural Network (GNN), successfully isolating compromised quantum fiber channels.

🧠 **Architecture**

⚡ **Simulation & Parsing**: Generates epochs of a 4-node diamond QKD topology (via SeQUeNCe), using high-speed C++ parsers to serialize raw telemetry into PyTorch graph tensors.

🕸️ **Edge-Centric GNN**: A custom neural network (model.py) that pools node representations to predict attack probabilities on specific directional fiber links ([8, 1] output).

🔑 **Cryptographic Loss**: A custom PyTorch loss function (key_rate_loss.py) that penalizes the network using actual key-rate degradation formulas, bridging ML with quantum cryptography.

🚀 **Quick Start**

1. **Install Dependencies**

pip install torch torch_geometric pandas



2. **Run the Pipeline**

# 1. Generate graph topologies (requires C++ compiler)
./parser.exe 

# 2. Train the Edge-Centric model
python train.py

# 3. Evaluate F1-scores against baseline thresholds
python evaluate.py



(Note: Check best_hparams.json and eval_summary.json for the latest evaluation metrics).

👨‍💻 **Team (CCBD Summer Internship)**

**Rishi**: Simulation & Data Parsing

**Avyukth**: Graph Neural Network & Training Pipeline

**Monish**: Quantum Crypto Math & Loss Function
