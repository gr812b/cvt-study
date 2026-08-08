# Revision 11 validation notes

The revision was checked for the following contracts before packaging:

- all overlay Python files compile;
- shared-event classification requires an event to be `shared` on every supported
  route before it receives a common contract;
- shared pass pooling selects one locally best compatible row per physical event
  and lap, rather than counting the same lap once per route;
- branch events remain route-specific;
- coarse/fine spacing cases adjust smoothing nodes so physical smoothing width
  stays near the nominal value;
- the normal study router selects the route-family robustness engine whenever the
  project has route variants/family handling enabled;
- the top-level robustness result is written under the normal `track_robustness`
  result family and contains a canonical HTML report plus a standard
  `track_ensemble_manifest.json`;
- the downstream ensemble manifest contains only robustness case IDs eligible on
  every route;
- uncertainty-side maximum-case truncation is route-balanced.

The full upstream repository test suite could not be executed in this container,
so the user should still treat the next Maryland robustness run as the integration
validation. The prior revision-10 Maryland run already demonstrated 16/16 successful
perturbations per route; revision 11 changes the shared gate contract, spacing-case
construction and orchestration/reporting around that working path.
