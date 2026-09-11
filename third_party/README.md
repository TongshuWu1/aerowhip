# Optional external dependencies

`natnet/PythonClient/` holds the locally installed NatNet Python client used by
the recording setup. Downloaded SDK files are ignored by Git. Use the SDK's own
installation and license instructions when setting up another machine.

Where a recorder asks for the directory containing `NatNetClient.py`, select
`third_party/natnet/PythonClient` relative to this checkout. These files are
external SDK code, not experiment measurements or trained models.
