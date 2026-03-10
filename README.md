# preframr
Predicting Register Encoded Frames Rendered As Musical Results

## overview

preframr is an experimental small language model that predicts [SID](https://en.wikipedia.org/wiki/MOS_Technology_6581) music from sequences of SID chip register calls intercepted in an emulator. This [example](example.mp3) output was predicted from an overfitted model trained with a small dataset including Goto80's [CBM_85](https://deepsid.chordian.net/?file=/MUSICIANS/G/Goto80/CBM_85.sid).

preframr's predicted register calls are rendered as audio via an emulated SID chip or a real SID chip via the ASID protocol (for example to an [Elektron SIDStation](https://en.wikipedia.org/wiki/Elektron_SidStation) or [Vessel](https://github.com/anarkiwi/vap) equipped C64). Because preframr works with low level SID register calls directly it can reproduce [subtle SID programming techniques](https://csdb.dk/release/?id=219545) such as combined waveforms and complex envelopes which cannot be represented as [standard MIDI files](https://midi.org/about-midi-part-4midi-files).

preframr is intended as a research tool for SID artists, explorers and historians. No pre-trained model is currently provided - you must obtain any and all necessary permission to use SID files for training.

## theory of operation

SID music on the C64 platform is traditionally realized as [SID files](https://www.preframr.c64.org/download/C64Music/DOCUMENTS/SID_file_format.txt). SID files are machine language programs for the C64 that play music by programming SID chip registers at regular intervals. These intervals are often derived from PAL or NTSC video frame rates which are approximately 50Hz or 60Hz respectively. While some SID music requires updates more frequently, the term "frame" is still used to represent a discrete set of register updates to be made at a specific cadence.

preframr predicts new frames from a list of frames as a prompt (for example, from another SID file). While preframr currently only supports single-SID music it could support other platforms that periodically generate control messages such low resolution video displays or speech chips.

preframr trains on SID register logs which are emitted by a modified VICE emulator. These logs contain timestamps for when interrupts were triggered and which SID register was updated with what value. This information is parsed into frames based on the detected interrupt interval, and then tokenized to represent each frame as a word compatible with a [conventional BPE style](https://en.wikipedia.org/wiki/Byte_pair_encoding) LLM tokenizer.

When inference is performed, predicted tokens are decoded back to register calls separated by interrupt intervals. 

## train your own model

* Gather a collection of SID files (for which you have permission) to use as training data
  
* Run the asid-vice tool to create SID register logs

```
$ mkdir /scratch/preframr/training-dumps
$ find /path/to/sid/files -name \*sid -print |parallel --progress docker run --rm -v /scratch:/scratch -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr/training-dumps --sid
```

* Train the model on the register logs

```
./retrain.sh
```

* Predict music

In this example, predict from a tune in the training set.
  
```
$ ./predict-nv.sh --temperature 1.1 --prompt-seq-len 128 --max-seq-len 8192 --start-n 0 --reglog /scratch/preframr/training-dumps/SOME_TUNE.dump.zst --wav /scratch/preframr/out.wav
```

## future work

* Scale training to very large SID datasets
* Multiple SID support
* Non-C64/SID platform support (e.g. ZX Spectrum)
* Model and tools for the identification of composer techniques from frames
* Tools to import/export frames from other applications (e.g. from and to SID Wizard)
* Real-time integration/workflows with DAWs
* Generate non-audio output (e.g. graphical VIC II output)

## acknowledgements

* [goto80](http://goto80.com) for his many creative achievements, encouragement and openness to exploration
* [Lucas Tindall](https://github.com/ltindall) for his machine learning mentoring
* [Jim Murphy](https://github.com/jimurphy) for his wisdom and support
