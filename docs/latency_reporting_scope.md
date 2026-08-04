# RD-VLA latency reporting scope

## Default scope

Unless a result explicitly states otherwise, latency improvements for the current optimization study refer to the **post-VLM action-policy path**. The VLM backbone forward pass is excluded.

This scope covers the modules executed after VLM hidden states are available, including the Prelude, recurrent core, Coda/output decode, output projection, warm-start state handling, scalar or latent stopping logic, and their runtime synchronization or wrapper overhead when included by the stated timer boundary.

## Reason

Both optimization families under study operate after the VLM backbone:

- SK1/midpoint warm-start reduces recurrent action-head computation by reusing a previous latent state.
- Coda scheduling removes repeated action decoding inside or after recurrent refinement.

Including the unchanged VLM backbone would dilute the measured effect and make comparisons depend on an unrelated backbone implementation.

## Required wording

Preferred terms:

- `post-VLM latency`
- `post-VLM action-policy latency`
- `action-head latency`, when the timer is exactly limited to the action head

Avoid using `end-to-end VLA latency` unless the VLM forward pass, preprocessing, and every other included component are actually timed.

Every result must still state its exact timer boundary. Excluding the VLM does not imply that all remaining runtime components are always included.
