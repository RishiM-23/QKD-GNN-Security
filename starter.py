import csv
import random
import math
from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel

def setup_network_topology(timeline):
    """
    Instantiates the 4-Node Diamond Topology in SeQUeNCe.
    Alice (A) connects to Relays (B, C), which connect to Bob (D).
    """
    nodes = {
        "Node_A": QuantumRouter("Node_A", timeline),
        "Node_B": QuantumRouter("Node_B", timeline),
        "Node_C": QuantumRouter("Node_C", timeline),
        "Node_D": QuantumRouter("Node_D", timeline)
    }

    # Define the links (Source, Destination, Distance in meters)
    topology_links = [
        ("Node_A", "Node_B", 10000), # 10 km
        ("Node_A", "Node_C", 12000), # 12 km
        ("Node_B", "Node_D", 15000), # 15 km
        ("Node_C", "Node_D", 11000)  # 11 km
    ]

    channels = {}

    for src, dst, dist in topology_links:
        link_id = f"Link_{src}_{dst}"
        
        # Quantum Channel for Qubits
        qc = QuantumChannel(f"qc_{src}_{dst}", timeline, distance=dist, attenuation=0.2)
        qc.set_ends(nodes[src], nodes[dst].name)
        
        # Classical Channel for Protocol Messages
        cc = ClassicalChannel(f"cc_{src}_{dst}", timeline, distance=dist)
        cc.set_ends(nodes[src], nodes[dst].name)

        channels[link_id] = {
            "src": src,
            "dst": dst,
            "qc": qc,
            "cc": cc,
            "distance": dist
        }

    return nodes, channels

def run_simulation_and_export_telemetry(timeline, channels, epochs=1000, output_file="sequence_telemetry_output.csv"):
    """
    Simulates the BB84 pulse transmissions across the network, 
    injects attacks, and exports the telemetry to CSV for the C++ parser.
    """
    headers = [
        "Link_ID", "QBER", "Signal_Count", "Decoy_Count", 
        "Attacked_Flag", "Key_Loss", "Source_Node"
    ]
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        print(f"Starting SeQUeNCe QKD Simulation for {epochs} epochs...")
        
        for epoch in range(epochs):
            timeline.time += 1e12 # Advance simulation time by 1 second per epoch

            for link_id, link_data in channels.items():
                src = link_data["src"]
                dist_km = link_data["distance"] / 1000.0
                
                # Base physical metrics based on fiber distance (0.2 dB/km attenuation)
                base_transmittance = 10 ** (-(0.2 * dist_km) / 10)
                
                # Baseline clean metrics
                signal_count = int(10000 * base_transmittance)
                decoy_count = int(2000 * base_transmittance)
                qber = random.uniform(0.01, 0.03) # 1% to 3% natural noise
                
                is_attacked = 0
                key_loss = 0.0

                # ---------------------------------------------------
                # THE ATTACK INJECTOR
                # ---------------------------------------------------
                # 5% chance an attacker injects Intercept-Resend or Blinding on a specific link
                if random.random() < 0.05:
                    is_attacked = 1
                    
                    # Attack Signatures: 
                    # 1. QBER spikes massively due to wavefunction collapse
                    qber += random.uniform(0.15, 0.25) 
                    
                    # 2. Decoy-to-Signal ratio skews because attacker blocks pulses
                    signal_count = int(signal_count * random.uniform(0.4, 0.6))
                    decoy_count = int(decoy_count * random.uniform(0.1, 0.3))
                    
                    # 3. Secret Key is destroyed
                    key_loss = random.uniform(0.8, 1.0) # 80% to 100% loss

                # Write the epoch data directly to the CSV
                writer.writerow([
                    link_id, 
                    round(qber, 4), 
                    signal_count, 
                    decoy_count, 
                    is_attacked, 
                    round(key_loss, 4), 
                    src
                ])

        print(f"Simulation Complete. Telemetry saved to {output_file}")

if __name__ == "__main__":
    # Initialize the SeQUeNCe Timeline
    sim_timeline = Timeline(1e12) # 1 second resolution baseline
    
    # Setup Topology
    network_nodes, network_channels = setup_network_topology(sim_timeline)
    
    # Run Simulation Loop
    run_simulation_and_export_telemetry(sim_timeline, network_channels, epochs=1000)