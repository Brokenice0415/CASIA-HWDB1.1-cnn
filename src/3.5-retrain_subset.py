#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

from keras.models import model_from_json
import json

import h5py

model_filepath = 'model-1612365213.json'
weights_filepath = 'weights-1612365213-0.921905.hdf5'
subset_filepath = 'HWDB1.1subset.hdf5'

with open(model_filepath) as f:
    d = json.load(f)
    model = model_from_json(json.dumps(d))

model.load_weights(weights_filepath)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

timestamp = int(time.time())
with open('model-%d.json' % timestamp, 'w') as f:
    d = json.loads(model.to_json())
    json.dump(d, f, indent=4)

with h5py.File(subset_filepath, 'r') as f:
    model.fit(f['trn/x'], f['trn/y'], validation_data=(f['vld/x'], f['vld/y']),
              epochs=10, batch_size=128, shuffle='batch', verbose=1)

    score = model.evaluate(f['tst/x'], f['tst/y'], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights('weights-%d-%f.hdf5' % (timestamp, score[1]))
