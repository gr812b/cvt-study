# Delivery manifest

This repository contains the complete measured-track CVT decision study:

- GPS cleaning, lap alignment, event metrics, speed-gate confidence, and simulator bundle export
- bounded and unbounded ideal-CVT simulation with measured braking gates
- separate wheel-speed and vehicle-speed host dynamics
- two-axis saturating tire model: peak traction and slip stiffness
- paired design sweeps and structural sensitivity studies
- concise method documentation, examples, and automated tests

Verified before packaging:

- 5 integrated simulator tests passed
- 11 GPS-analysis tests passed
- standard measured-track smoke run passed reference dominance
