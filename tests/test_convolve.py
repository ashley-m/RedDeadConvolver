import unittest

from reddeadconvolver.convolve import ConvolutionConfig, convolve_signals


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


if __name__ == "__main__":
    unittest.main()
