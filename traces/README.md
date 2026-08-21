# Trace files

Every trace here was **recorded from armulator's own models**, not from hardware. They exist to catch regressions: if a change alters peripheral behaviour, replaying these will diverge.

They do **not** validate the models against real silicon. A replay of a self-recorded trace is circular by construction, and the replay report says so in its output.

To validate against hardware, capture a real trace on a Pi and replay that instead. See `RASPI.md` for the ftrace recipe.

Regenerate with `python3 tools/record_baselines.py`.
