"""Capture, VAD, and utterance segmentation (FR-01).

Planned adapters, both satisfying :class:`tpdk_overlay.ports.AudioSource`:

* ``capture.py`` - WASAPI loopback or microphone via ``soundcard``, 16 kHz mono.
* ``vad.py`` - Silero VAD state machine emitting ``SpeechStart``/``SpeechEnd``
  with pre-roll and a max-utterance cut.
"""
