"""Optional, synthesized renderer-only interface sounds."""

from __future__ import annotations

import math

import numpy as np
import pygame


class UiSoundEffects:
    """Small generated UI chimes; unavailable audio devices fail silently."""

    def __init__(self, enabled: bool = True, sample_rate: int = 22050) -> None:
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.channels = 1
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._music: pygame.mixer.Sound | None = None
        self._music_channel: pygame.mixer.Channel | None = None
        self.available = False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
            mixer_info = pygame.mixer.get_init()
            if mixer_info:
                self.sample_rate, _, self.channels = mixer_info
            self._sounds = {
                "tap": self._tone((520,), 0.045, 0.18),
                "open": self._tone((587, 784), 0.12, 0.18),
                "start": self._tone((523, 659, 784), 0.20, 0.20),
            }
            self._music = self._make_ambient_music()
            self.available = True
        except pygame.error:
            self.available = False

    def _tone(self, notes: tuple[int, ...], duration: float, volume: float):
        count = max(1, int(self.sample_rate * duration))
        t = np.arange(count, dtype=np.float32) / self.sample_rate
        envelope = np.minimum(1.0, t * 100) * np.exp(-t * 9)
        signal = np.zeros(count, dtype=np.float32)
        for index, note in enumerate(notes):
            signal += np.sin(2 * math.pi * note * t) / (index + 1)
        signal *= envelope * volume / max(1, len(notes))
        pcm = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        if self.channels > 1:
            pcm = np.repeat(pcm[:, None], self.channels, axis=1)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def _make_ambient_music(self):
        """Create a soft, seamless 16-second chord-and-bell loop in memory."""
        duration = 16.0
        count = int(self.sample_rate * duration)
        t = np.arange(count, dtype=np.float32) / self.sample_rate
        signal = np.zeros(count, dtype=np.float32)
        chords = (
            (130.81, 164.81, 196.00),
            (110.00, 130.81, 164.81),
            (87.31, 110.00, 130.81),
            (98.00, 123.47, 146.83),
        )
        for index, notes in enumerate(chords):
            local = t - index * 4.0
            active = (local >= 0.0) & (local < 4.0)
            envelope = np.clip(local / 0.75, 0.0, 1.0) * np.clip((4.0 - local) / 0.85, 0.0, 1.0)
            pad = sum(np.sin(2 * math.pi * note * t) for note in notes) / len(notes)
            signal += pad * envelope * active * 0.16
        melody = (392.00, 523.25, 440.00, 329.63, 392.00, 587.33, 523.25, 440.00)
        for index, note in enumerate(melody):
            local = t - index * 2.0
            active = (local >= 0.0) & (local < 2.0)
            envelope = np.clip(local / 0.08, 0.0, 1.0) * np.exp(-np.maximum(local, 0.0) * 2.6)
            bell = np.sin(2 * math.pi * note * t) + 0.22 * np.sin(2 * math.pi * note * 2.01 * t)
            signal += bell * envelope * active * 0.055
        signal *= 0.78
        pcm = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        if self.channels > 1:
            pcm = np.repeat(pcm[:, None], self.channels, axis=1)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def start_music(self) -> None:
        if self.enabled and self.available and self._music is not None:
            self._music_channel = self._music.play(loops=-1)
            if self._music_channel is not None:
                self._music_channel.set_volume(0.20)

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if self.enabled:
            self.start_music()
        elif self._music_channel is not None:
            self._music_channel.stop()
            self._music_channel = None
        return self.enabled

    def play(self, name: str = "tap") -> None:
        if self.enabled and self.available:
            sound = self._sounds.get(name)
            if sound is not None:
                sound.play()
