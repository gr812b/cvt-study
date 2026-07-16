# Canonical terminology

These terms are contracts, not interchangeable prose.

- **Project:** one self-contained study workspace with a track, one or more vehicles,
  study definitions, and local results.
- **Track:** the physical course and its track-specific environment/model inputs.
- **Run:** one GPX recording associated with one vehicle and, where known, one driver.
- **Lap:** one detected traversal of the closed course within a run.
- **Centreline:** the common spatial reference reconstructed from selected runs.
- **Track coordinate `s`:** distance along the centreline, periodic over one lap.
- **Physical feature:** one real turn, obstacle, rough patch, or other bounded feature.
- **Response group:** one or more physical features whose GPS response cannot be
  separated reliably. Grouping does not erase the individual physical features.
- **Start:** first physical contact or beginning of a modeled feature interval.
- **Anchor:** a recognizable map/video reference used to locate a feature.
- **End:** end of the physical feature interval.
- **Entry measurement point:** computed upstream location used to measure approach response.
- **Minimum-speed point:** observed local minimum associated with the response group.
- **Recovery point:** computed downstream location where acceleration/recovery is assessed.
- **Gate candidate:** an evidence-derived possible driver-limited speed ceiling.
- **Accepted gate:** a candidate allowed to constrain simulation under the selected policy.
- **Review-only gate:** retained in evidence outputs but excluded from default simulation.
- **Obstacle model:** explicit equation mapping a feature and scenario to force or energy.
- **Nominal value:** central reported value, not an assertion of exactness.
- **Uncertainty:** declared spread or discrete alternatives around an input or model choice.
- **Fixed value:** an explicitly justified zero-uncertainty assumption.
- **Measured-track variability:** repeatable variation inferred from GPX/event evidence.
- **Structural uncertainty:** uncertainty in vehicle, terrain, and model parameters.
- **Monte Carlo estimation error:** uncertainty in estimated statistics due to finite samples.
- **Scenario:** one complete, paired realization of all sampled inputs.
- **Design:** one candidate drivetrain configuration.
- **Unbounded reference:** the same scenario and vehicle with idealized unlimited CVT ratio.
