import os
import unittest

from llama_zip import LlamaZip


class RoundTripTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("SMOLLM2_135M_GGUF"), "requires a SmolLM2-135M GGUF")
    def test_long_whitespace_token(self):
        codec = LlamaZip(os.environ["SMOLLM2_135M_GGUF"], n_ctx=64, n_gpu_layers=0)
        # SmolLM2 encodes a newline followed by 48 spaces as one 49-byte token.
        original = b"\r\n" + b" " * 54 + b"sea!"
        self.assertEqual(codec.decompress(codec.compress(original)), original)
