# Consensus cleanup drop-in v3

Fixes `assignment destination is read-only` under Pandas Copy-on-Write by
creating explicit writable NumPy copies before consensus arrays are modified.

Extract directly over the repository root, reinstall editable dependencies, and
run the tests/build again.
