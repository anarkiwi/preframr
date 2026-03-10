```
 _ __  _ __ ___ / _|_ __ __ _ _ __ ___  _ __
| '_ \| '__/ _ \ |_| '__/ _` | '_ ` _ \| '__|
| |_) | | |  __/  _| | | (_| | | | | | | |
| .__/|_|  \___|_| |_|  \__,_|_| |_| |_|_|
|_|

```

Predicting Register Encoded Frames Rendered As Musical Results

## overview

preframr is an experimental small language model that predicts [SID](https://en.wikipedia.org/wiki/MOS_Technology_6581) music from sequences of SID chip register calls intercepted in an emulator. This [example](example.mp3) output was predicted from an overfitted model trained with a small dataset including Goto80's [CBM_85](https://deepsid.chordian.net/?file=/MUSICIANS/G/Goto80/CBM_85.sid).

preframr's predicted register calls are rendered as audio via an emulated SID chip or a real SID chip via the ASID protocol (for example to an [Elektron SIDStation](https://en.wikipedia.org/wiki/Elektron_SidStation) or [Vessel](https://github.com/anarkiwi/vap) equipped C64). Because preframr works with low level SID register calls directly it can reproduce [subtle SID programming techniques](https://csdb.dk/release/?id=219545) such as combined waveforms and complex envelopes which cannot be represented as [standard MIDI files](https://midi.org/about-midi-part-4midi-files) or MIDI tokenizers such as [MidiTok](https://github.com/Natooz/MidiTok). While register calls are less complex than audio samples they are by definition musically expressive as they realize all C64 chipmusic and avoid the need to process audio samples with frameworks such as [Magenta](https://github.com/magenta/magenta-realtime). preframr's register call logging could be directly used with other register logging formats such as [VGMusic](https://en.wikipedia.org/wiki/VGM_(file_format)). While preframr currently only supports single-SID music it could support other platforms that periodically generate control messages such low resolution video displays or speech chips.

## why?

preframr is intended as a research tool for SID artists, explorers and historians, at a scale large enough to be useful artistically (for example, to assist artists in the creation of their music), but small enough to be within the limits of consumer grade GPUs. This enables an individual to manage their own "foundation" model. Therefore, no pre-trained model is currently provided - you must obtain any and all necessary permission to use SID files for training.

## theory of operation

### parsing

SID music on the C64 platform is traditionally realized as [SID files](https://www.hvsc.c64.org/download/C64Music/DOCUMENTS/SID_file_format.txt). SID files are machine language programs for the C64 that play music by programming SID chip registers at regular intervals. These intervals are often derived from PAL or NTSC video frame rates which are approximately 50Hz or 60Hz respectively. While some SID music requires updates more frequently, the term "frame" is still used to represent a discrete set of register updates to be made at a specific cadence.

preframr predicts new frames from a sequence of frames as a prompt (for example, parsed from some SID file). preframr trains on SID register logs which are emitted by a modified VICE emulator. These logs contain timestamps for when interrupts were triggered and which SID register was updated with what value. This difference information is parsed into frames based on the detected interrupt interval, and then tokenized to represent each frame as a word compatible with [BPE or Unigram](https://huggingface.co/docs/transformers/tokenizer_summary) tokenization. When inference is performed, predicted tokens are decoded back to register calls separated by interrupt intervals.

### complexity reduction and tokenization mechanics

As with all LLM tokenizers preframr must balance compression efficiency and fidelity in encoding musical information. The SID chip has three voices which are in principle interchangeable (with some restrictions such as when ring or sync modulation features are used in which specific voice registers must be programmed together). As these voices are scarce resources SID musicians closely manage them and the role a given voice plays such as rendering a percussion sound or a melodic lead can change often, potentially multiple times per second. preframr must therefore encode musical information agnostic to an individual voice allocation. SID voice frequency, filter cutoff and PCM modulation are greater than 8 bit resolution which while expressive are expensive to encode.

First, preframr applies configurable bit resolution reduction and register combination in frequency, filter cutoff, and PCM modulation parameters (for example, SID voice 0 frequency is programmed via 8 bit registers 0 and 1, which are combined into one register 0), Then preframr applies the following complexity reduction strategies using virtual register calls (where the register value is less than 0) to perform specific functions:

* register value redundancy
    * remove redundant control register values (for example, PCM modulation changes removed where pulse waveform is not selected)
* voice registers and augmentation
    * each training sequence sample is rendered 3 times with each voice rotated to assume each original voice assignment
    * register programming order within a frame is normalized in voice by voice order, sorted by current voice frequency and control register values (ensuring intra-frame register order is consistent and comparable between sequences and can be tokenized to the same token)
    * virtual register FRAME_REG encodes the high bits of current frequency assignments and the order of voice registers with a frame (retaining beyond context frequency information)
    * virtual register VOICE_REG signifies voice selection within a frame
* change opcodes (encoding common SID music ornamentation techniques)
    * only register calls where the value changes between frames are used (SET_OP)
    * small changes (for example, within two octaves for frequency) are encoded with DIFF_OP
    * alternating changes every frame (alternately adding or subtracting a fixed value from another fixed value) are encoded with FLIP_OP
    * repeating changes every frame (repeatedly adding or subtracting a fixed value from another, such as a PCM modulation sweep up or down) are encoded with REPEAT_OP
  
### tokenization example

This example shows how CBM_85 is tokenized, from this original partial register log starting from the first frame:

```
>>> import pandas as pd
>>> df = pd.read_parquet("CBM_85.None.dump.parquet")
>>> df[:50]
     clock     irq  chipno  reg  val
0      342       0       0   24   15
1    20202   19592       0   16  153
2    20211   19592       0   17    1
3    20219   19592       0   22    4
4    20228   19592       0   14   24
5    20237   19592       0   15    1
6    20246   19592       0   19   68
7    20255   19592       0   20   94
8    20268   19592       0   18   64
9    20280   19592       0   24   31
10   20808   19592       0    9  153
11   20817   19592       0   10    1
12   20834   19592       0    7   23
13   20843   19592       0    8    1
14   20852   19592       0   12   68
15   20861   19592       0   13   94
16   20874   19592       0   11   64
17   21412   19592       0    2  153
18   21421   19592       0    3    1
19   21438   19592       0    0   22
20   21447   19592       0    1    1
21   21465   19592       0    6  249
22   21478   19592       0    4   32
...
```

This is the equivalent parsed output. Register -128 is FRAME_REG, delineating a frame, and -126 is VOICE_REG signifying moving to the next voice. FRAME_REG's value encodes the voice order within the frame. In this case voice 0 comes first, and its sustain/release register (6) is set to 249 at line 4 using op 0 which is SET_OP. This is equivalent to line 21 in the original log. Voice 0's frequency register is increased (op 1, DIFF_OP) by a value of 24 from 0. As this is the first frame, the difference from 0 is small and normalized from the original value 22 at line 19 in the original log.

```
>>> df = pd.read_parquet("CBM_85.1.0.parquet")
>>> df[:50]
     reg   val   diff    irq  op
0   -128    57  19656  19656   0
1      0    24     32  19656   1
2      2    25     32  19656   1
3      4    32     32  19656   0
4      6   249     32  19656   0
5   -128     0     32  19656   0
6      0    24     32  19656   1
7      2    25     32  19656   1
8      4    64     32  19656   0
9      5    68     32  19656   0
10     6    94     32  19656   0
11  -128     0     32  19656   0
12     0    24     32  19656   1
13     2    25     32  19656   1
14     4    64     32  19656   0
15     5    68     32  19656   0
16     6    94     32  19656   0
17    21   256     32  19656   0
18    24    31     32  19656   0
```

## train your own model

A complete example of training and predicting music based on Goto80's music is provided. It is assumed docker and NVIDIA drivers have already been installed.

```
$ ./build.sh
$ ./run_int_test.sh
```

This will cause all needed containers to be built, several SID tunes to be parsed and tokenized, and a model trained which should take a few minutes on an RTX 4090 GPU. Several predictions from the training set will be made which should be overfitted (accuracy 1).

## using preframr creatively

Currently, preframr is a stand alone framework that only supports inference from SID files it has parsed, and predicting the results as new sequences and/or audio output. It can of course predict new sequences from sequences it has not been trained on. Work to integrate preframr with DAW tools is planned.

## future work

* Tools to import/export frames from other applications (e.g. from and to SID Wizard)
* Real-time integration/workflows with DAWs
* Multiple SID support
* Non-C64/SID platform support (e.g. ZX Spectrum)
* Model and tools for the identification of composer techniques from frames
* Generate non-audio output (e.g. graphical VIC II output)

## acknowledgements

* [goto80](http://goto80.com) for his many creative achievements, encouragement and openness to exploration
* [Lucas Tindall](https://github.com/ltindall) for his machine learning mentoring
* [Jim Murphy](https://github.com/jimurphy) for his wisdom and support
