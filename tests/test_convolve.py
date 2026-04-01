import unittest
from unittest import mock

from reddeadconvolver.convolve import ConvolutionConfig, convolve_signals, load_audio


class ConvolveTests(unittest.TestCase):
    def test_mono_full_convolution_expected_values(self):
        signal = [1.0, 2.0, 3.0]
        ir = [1.0, 0.5]

        out = convolve_signals(signal, ir, ConvolutionConfig(mode="full", normalize="none"))
        self.assertEqual(out, [1.0, 2.5, 4.0, 1.5])

    def test_stereo_match_strategy_uses_each_ir_channel(self):
        signal = [[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]
        ir = [[1.0, 0.0], [0.5, 1.0]]

        out = convolve_signals(signal, ir, ConvolutionConfig(mode="full", normalize="none"))
        self.assertEqual(
            out,
            [
                [1.0, 0.0],
                [0.5, 2.0],
                [1.0, 1.0],
                [0.5, 0.0],
            ],
        )

    def test_stereo_signal_mono_ir_is_duplicated_in_match_mode(self):
        signal = [[1.0, 0.0], [0.5, 0.5]]
        mono_ir = [1.0, -1.0]

        out = convolve_signals(
            signal,
            mono_ir,
            ConvolutionConfig(mode="full", normalize="none", channel_strategy="match"),
        )

        self.assertEqual(
            out,
            [
                [1.0, 0.0],
                [-0.5, 0.5],
                [-0.5, -0.5],
            ],
        )


    def test_fft_and_direct_methods_match(self):
        signal = [0.2, -0.3, 0.5, 0.1]
        ir = [1.0, 0.25, -0.1]

        fft_out = convolve_signals(signal, ir, ConvolutionConfig(mode="full", normalize="none", method="fft"))
        direct_out = convolve_signals(signal, ir, ConvolutionConfig(mode="full", normalize="none", method="direct"))

        self.assertEqual(len(fft_out), len(direct_out))
        for fft_sample, direct_sample in zip(fft_out, direct_out):
            self.assertAlmostEqual(fft_sample, direct_sample, places=6)

    def test_default_method_is_fft(self):
        cfg = ConvolutionConfig()
        self.assertEqual(cfg.method, "fft")

    def test_load_audio_uses_soundfile_for_non_wav(self):
        fake_sf = mock.Mock()
        fake_sf.read.return_value = ([[0.1], [-0.2], [0.3]], 48000)

        with mock.patch.dict("sys.modules", {"soundfile": fake_sf}):
            audio, sample_rate = load_audio("test.mp3")

        self.assertEqual(audio, [0.1, -0.2, 0.3])
        self.assertEqual(sample_rate, 48000)


if __name__ == "__main__":
    unittest.main()
