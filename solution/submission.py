import numpy as np
import soundfile as sf


class EnergyDetector:
    """
    Simple baseline detector based on short-time energy.

    Training:
      - estimate an energy threshold from labeled train clips

    Prediction:
      - detect frames whose energy is above the threshold
      - merge consecutive active frames into segments
    """

    def __init__(self, frame_size=1024, hop_size=512, min_duration=0.1):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_duration = min_duration
        self.threshold = None

    def _load_audio(self, audio_path):
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        return audio, sr

    def _frame_energies(self, audio):
        if len(audio) < self.frame_size:
            pad = np.zeros(self.frame_size, dtype=np.float32)
            pad[: len(audio)] = audio
            audio = pad

        energies = []
        starts = []

        for i in range(0, len(audio) - self.frame_size + 1, self.hop_size):
            frame = audio[i : i + self.frame_size]
            energy = float(np.mean(frame**2))
            energies.append(energy)
            starts.append(i)

        return np.array(energies), np.array(starts)

    def _interval_overlaps(self, frame_start, frame_end, segments, sr):
        """
        Return True if the frame overlaps at least one labeled emergency segment.
        """
        t0 = frame_start / sr
        t1 = frame_end / sr

        for seg_start, seg_end in segments:
            inter = max(0.0, min(t1, seg_end) - max(t0, seg_start))
            if inter > 0:
                return True
        return False

    def fit(self, train_features, train_labels, data_dir):
        """
        Estimate the energy threshold from the training set.
        """
        train_labels = train_labels.copy()
        train_labels["sample_id"] = train_labels["sample_id"].astype(str)

        label_map = {}
        for sample_id, g in train_labels.groupby("sample_id"):
            label_map[str(sample_id)] = list(
                zip(g["start"].astype(float), g["end"].astype(float))
            )

        pos_energies = []
        neg_energies = []

        for row in train_features.itertuples(index=False):
            sample_id = str(row.sample_id)
            audio_path = data_dir / row.audio_path

            try:
                audio, sr = self._load_audio(audio_path)
            except Exception:
                continue

            energies, frame_starts = self._frame_energies(audio)
            segments = label_map.get(sample_id, [])

            for energy, start_sample in zip(energies, frame_starts):
                end_sample = start_sample + self.frame_size
                if self._interval_overlaps(start_sample, end_sample, segments, sr):
                    pos_energies.append(energy)
                else:
                    neg_energies.append(energy)

        pos_mean = float(np.mean(pos_energies))
        neg_mean = float(np.mean(neg_energies))

        # Midpoint between average positive and negative frame energy
        self.threshold = 0.5 * (pos_mean + neg_mean)
        return self

    def predict(self, audio_path):
        """
        Return a list of segments:
          [{"start": ..., "end": ...}, ...]
        """
        if self.threshold is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        audio, sr = self._load_audio(audio_path)
        energies, frame_starts = self._frame_energies(audio)

        active = energies > self.threshold

        segments = []
        start_idx = None

        for i, is_active in enumerate(active):
            if is_active and start_idx is None:
                start_idx = i
            elif not is_active and start_idx is not None:
                end_idx = i
                start_time = frame_starts[start_idx] / sr
                end_sample = frame_starts[end_idx - 1] + self.frame_size
                end_time = end_sample / sr

                if end_time - start_time >= self.min_duration:
                    segments.append(
                        {
                            "start": float(start_time),
                            "end": float(end_time),
                        }
                    )
                start_idx = None

        # Close last active segment
        if start_idx is not None:
            start_time = frame_starts[start_idx] / sr
            end_sample = frame_starts[len(active) - 1] + self.frame_size
            end_time = min(len(audio) / sr, end_sample / sr)

            if end_time - start_time >= self.min_duration:
                segments.append(
                    {
                        "start": float(start_time),
                        "end": float(end_time),
                    }
                )

        return segments


def get_model():
    return EnergyDetector()
