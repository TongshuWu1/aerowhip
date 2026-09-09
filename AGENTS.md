# Simulator branch - complete application, no historical data

Read HANDOFF.md first. The user explicitly wants the WHOLE application/UI and
all training, policy, model, fitting, rehearsal/export, recording and diagnostic
code. Do not reduce this to a standalone physics viewer or remove app features.

This is a fresh source-only branch. Do not import collected real-flight data,
old fitted M0/M1, NN weights, trained policy checkpoints, or historical outputs.
Keep model/policy/data libraries empty until new simulator experiments produce
artifacts. The nominal example profile is unfitted and must not be called M0.

Next: implement an independent Isaac Lab/PhysX drone + passive cable plant,
collect preliminary data there, and fit a NEW M0. Existing Isaac Lab hosting
uses our external model; it is not the independent PhysX plant. Keep hidden
plant parameters separate from the model being fitted. Preserve the 30 Hz
FullState command contract and plan for 100 Hz synthetic measurements.

Do not launch fits or training without user authorization. Future fits should
stop on practical convergence/plateau and use essential validation. The inherited
five-take adaptation runner needs preliminary/synthetic-data integration before
M0 fitting. Keep existing code available, but do not claim this is already done.
