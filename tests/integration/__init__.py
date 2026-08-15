"""The integration tier.

An integration test constructs real components through `build_perception` and
asserts on behaviour crossing at least two module boundaries. It may use a
fake **model**, because model output is nondeterministic and this project has
never asserted on model text. It may not use a fake **source, detector,
hasher, store, or registry** — those are the seams under test.
"""
