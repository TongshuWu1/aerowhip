"""Transport-neutral logging sink for ROS callbacks or a standalone controller.

No subscribers, publishers, flight commands, clock inference or frame conversion
live here. The caller provides explicitly normalized tracking and force values.
"""
import csv
import json
from pathlib import Path
import threading
from experimental_data.flight_trials import template
from experimental_data.io import atomic_json,canonical_json_hash


class FlightRecorder:
    def __init__(self,directory,model,task,*,metadata=None):
        self.directory=Path(directory);template(self.directory)
        atomic_json(self.directory/'model.json',model);atomic_json(self.directory/'task.json',task)
        self.metadata=json.loads((self.directory/'trial.json').read_text())
        self.metadata.update(metadata or {})
        self.metadata['model_sha256']=canonical_json_hash(model)
        atomic_json(self.directory/'trial.json',self.metadata)
        self.lock=threading.Lock();self.closed=False
        self.tracking_stream=(self.directory/'tracking.csv').open('a',newline='')
        self.command_stream=(self.directory/'commands.csv').open('a',newline='')
        self.controller_stream=(self.directory/'controller_imu.jsonl').open('a',encoding='utf-8')
        self.tracking_writer=csv.writer(self.tracking_stream);self.command_writer=csv.writer(self.command_stream)

    def _check_open(self):
        if self.closed:raise ValueError('This flight recorder is closed')

    def record_tracking(self,time_s,drone_position_m,body_to_world_xyzw,markers_m,*,drone_valid=True,marker_valid=None):
        if len(drone_position_m)!=3 or len(body_to_world_xyzw)!=4 or len(markers_m)!=10 or any(len(p)!=3 for p in markers_m):
            raise ValueError('Require drone XYZ, quaternion XYZW, and ten XYZ cable markers')
        valid=[True]*10 if marker_valid is None else marker_valid
        if len(valid)!=10:raise ValueError('Require ten marker validity flags')
        row=[time_s,*drone_position_m,*body_to_world_xyzw,int(drone_valid)]
        for point,flag in zip(markers_m,valid):row.extend([*point,int(flag)])
        with self.lock:
            self._check_open();self.tracking_writer.writerow(row);self.tracking_stream.flush()

    def record_sent_force(self,time_s,force_world_n):
        if len(force_world_n)!=3:raise ValueError('Require an XYZ force in newtons')
        with self.lock:
            self._check_open();self.command_writer.writerow([time_s,*force_world_n]);self.command_stream.flush()

    def record_controller(self,time_s,payload,*,clock,frame):
        # Retain native timing/frame and complete caller-provided fields. These
        # measurements are not silently interpreted as applied world force.
        value=json.dumps(dict(time_s=time_s,clock=clock,frame=frame,data=payload),allow_nan=False)
        with self.lock:
            self._check_open();self.controller_stream.write(value+'\n');self.controller_stream.flush()

    def close(self,**reviewed_metadata):
        with self.lock:
            if self.closed:return
            for stream in (self.tracking_stream,self.command_stream,self.controller_stream):stream.close()
            self.metadata.update(reviewed_metadata)
            atomic_json(self.directory/'trial.json',self.metadata);self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
