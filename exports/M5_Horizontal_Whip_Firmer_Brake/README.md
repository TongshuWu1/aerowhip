# M5_Horizontal_Whip_Firmer_Brake

Use fullstate_30hz.csv with the existing controller and launch command.
Initial tracked-origin position: (0.0000, 0.0000, 1.2884) m.
Selected simulated release point: (3.0947, 0.9717, 1.4500) m.
Complete command duration: 13.300 s, including braking, return and final hold.
Braking duration: 1.000 s.
Execute the complete CSV. Its first row is the required starting hover, not a takeoff command. The two exports have different starting heights.

Save recordings under flight_take/M5. Each export has its own recording folder so the two tasks remain separate.
Full model-based recovery checks passed; this does not establish physical contact or measured flight performance. Frozen prediction and model remain in the source rehearsal listed in export.json.
CSV SHA-256: 1ab8e1ed748f3b755958aa4730ca2df625b17b171727831a4b261a6a50053925
