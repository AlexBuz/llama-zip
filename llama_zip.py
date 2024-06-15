import argparse
import codecs
import signal
import sys
from collections import deque

import numpy as np
from llama_cpp import Llama
from more_itertools import consume
from tqdm import tqdm


NUM_STATE_BITS = 64
FREQ_SCALE_FACTOR = 1 << 32
BASE64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


class ArithmeticCoderBase:
    def __init__(self):
        full_range = 1 << NUM_STATE_BITS
        self.half_range = full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, cum_freqs, symbol):
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        symhigh = int(cum_freqs[symbol])
        self.high = self.low + symhigh * range // total - 1
        symlow = int(cum_freqs[symbol - 1]) if symbol > 0 else 0
        self.low = self.low + symlow * range // total
        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = (self.low << 1) & self.state_mask
            self.high = ((self.high << 1) & self.state_mask) | 1
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


class Encoder(ArithmeticCoderBase):
    def __init__(self):
        super().__init__()
        self.encoded_data = []
        self.num_underflow = 0

    def get_encoded(self):
        return self.encoded_data

    def encode_symbol(self, cum_freqs, symbol):
        self.update(cum_freqs, symbol)

    def finish(self):
        self.encoded_data.append(1)

    def shift(self):
        bit = self.low >> (NUM_STATE_BITS - 1)
        self.encoded_data.append(bit)
        self.encoded_data.extend([bit ^ 1] * self.num_underflow)
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1


class Decoder(ArithmeticCoderBase):
    def __init__(self, encoded_bits):
        super().__init__()
        self.input = deque(encoded_bits)
        self.code = sum(
            self.read_code_bit() << i for i in range(NUM_STATE_BITS - 1, -1, -1)
        )

    def decode_symbol(self, cum_freqs):
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
        symbol = np.searchsorted(cum_freqs, value, side="right")
        self.update(cum_freqs, symbol)
        return symbol

    def shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

    def underflow(self):
        self.code = (
            (self.code & self.half_range)
            | ((self.code << 1) & (self.state_mask >> 1))
            | self.read_code_bit()
        )

    def read_code_bit(self):
        return self.input.popleft() if len(self.input) else 0


# Based on Rust's std::str::Utf8Chunks
class Utf8Chunks:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        if not self.source:
            raise StopIteration

        TAG_CONT_U8 = 128

        def safe_get(xs, i):
            try:
                return xs[i]
            except IndexError:
                return 0

        i = 0
        valid_up_to = 0
        while i < len(self.source):
            byte = self.source[i]
            i += 1

            if byte < 0x80:
                # ASCII
                pass
            elif 0xC2 <= byte <= 0xDF:
                # 2-byte sequence
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xE0 <= byte <= 0xEF:
                # 3-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xE0 == byte and 0xA0 <= next_byte <= 0xBF:
                    pass
                elif 0xE1 <= byte <= 0xEC and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xED == byte and 0x80 <= next_byte <= 0x9F:
                    pass
                elif 0xEE <= byte <= 0xEF and 0x80 <= next_byte <= 0xBF:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xF0 <= byte <= 0xF4:
                # 4-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xF0 == byte and 0x90 <= next_byte <= 0xBF:
                    pass
                elif 0xF1 <= byte <= 0xF3 and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xF4 == byte and 0x80 <= next_byte <= 0x8F:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            else:
                break

            valid_up_to = i

        inspected, remaining = self.source[:i], self.source[i:]
        self.source = remaining

        valid, invalid = inspected[:valid_up_to], inspected[valid_up_to:]
        return valid, invalid


PUA_START = 0xE000


def bytes_to_utf8(data: bytes):
    output = bytearray()
    for valid, invalid in Utf8Chunks(data):
        for char in valid.decode("utf-8"):
            if PUA_START <= ord(char) <= PUA_START + 0xFF:
                for byte in char.encode("utf-8"):
                    output.extend(chr(PUA_START + byte).encode("utf-8"))
            else:
                output.extend(char.encode("utf-8"))
        for byte in invalid:
            output.extend(chr(PUA_START + byte).encode("utf-8"))
    return bytes(output)


def utf8_to_bytes(data: str):
    output = bytearray()
    for char in data:
        if PUA_START <= ord(char) <= PUA_START + 0xFF:
            output.append(ord(char) - PUA_START)
        else:
            output.extend(char.encode("utf-8"))
    return bytes(output)


class LlamaZip:
    def __init__(
        self, model_path, n_ctx=0, n_gpu_layers=-1, use_mlock=False, verbose=False
    ):
        self.verbose = verbose
        self.load_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
        )

    def load_model(self, model_path, n_ctx, n_gpu_layers, use_mlock):
        loading_message = "Loading model..."
        if self.verbose:
            print(loading_message, end="", flush=True, file=sys.stderr)
        self.model = Llama(
            model_path=model_path,
            use_mlock=use_mlock,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        if self.verbose:
            print(
                "\r" + " " * len(loading_message) + "\r",
                end="",
                flush=True,
                file=sys.stderr,
            )

    def compute_cdf(self, logits):
        logprobs = self.model.logits_to_logprobs(logits)
        probs = np.exp(logprobs)
        freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
        cum_freqs = np.cumsum(freqs)
        return cum_freqs

    def compress(self, uncompressed: bytes, window_overlap=0) -> bytes:
        def bits_to_base64(bits):
            while bits and bits[-1] == 0:
                bits.pop()
            while len(bits) % 6 != 0:
                bits.append(0)
            return bytes(
                BASE64[int("".join(str(bit) for bit in bits[i : i + 6]), 2)]
                for i in range(0, len(bits), 6)
            )

        def sigint_handler(*_):
            nonlocal interrupted
            interrupted = True

        def process_logits(_, logits):
            nonlocal next_token_idx
            if interrupted and next_token_idx < len(tokens) - 1:
                next_token_idx = len(tokens) - 1
                if self.verbose:
                    print(file=sys.stderr)
            next_token = tokens[next_token_idx]
            next_token_idx += 1
            cdf = self.compute_cdf(logits)
            token_encoder.encode_symbol(cdf, next_token)
            progress_bar.update()
            logits[next_token] = np.inf
            return logits

        def should_stop(tokens_so_far, logits):
            return (
                np.argmax(logits) == self.model.token_eos()
                or len(tokens_so_far) == self.model.n_ctx()
            )

        self.model.reset()
        tokens = self.model.tokenize(bytes_to_utf8(uncompressed), add_bos=False)
        tokens.append(self.model.token_eos())
        next_token_idx = 0
        token_encoder = Encoder()

        interrupted = False
        s = signal.signal(signal.SIGINT, sigint_handler)

        progress_bar = tqdm(
            total=len(tokens),
            mininterval=1 / 30,
            desc="Compressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
            disable=not self.verbose,
        )

        while next_token_idx < len(tokens):
            start_idx = max(0, next_token_idx - window_overlap)
            consume(
                self.model.generate(
                    tokens=[self.model.token_bos()] + tokens[start_idx:next_token_idx],
                    temp=0.0,
                    logits_processor=process_logits,
                    stopping_criteria=should_stop,
                )
            )
        progress_bar.close()

        token_encoder.finish()
        compressed = bits_to_base64(token_encoder.get_encoded())
        if self.verbose:
            sys.stdout.buffer.write(compressed)
            sys.stdout.buffer.flush()

        signal.signal(signal.SIGINT, s)

        return compressed

    def tokenizer_adds_space_prefix(self):
        space = b" "
        double_space = b"  "
        tokenized = self.model.tokenize(space, add_bos=False)
        return self.model.detokenize(tokenized) == double_space

    def decompress(self, compressed: bytes, window_overlap=0) -> bytes:
        def base64_to_bits(string):
            bits = [int(bit) for char in string for bit in f"{BASE64.index(char):06b}"]
            return bits

        def process_logits(_, logits):
            cdf = self.compute_cdf(logits)
            next_token = token_decoder.decode_symbol(cdf)
            logits[next_token] = np.inf
            if next_token == self.model.token_eos():
                return logits
            next_utf8 = self.model.detokenize([next_token])
            if (
                len(seen_tokens) == 0
                and next_utf8.startswith(b" ")
                and self.tokenizer_adds_space_prefix()
            ):
                next_utf8 = next_utf8[1:]
            seen_tokens.append(next_token)
            next_bytes = utf8_to_bytes(utf8_decoder.decode(next_utf8))
            decompressed.extend(next_bytes)
            if self.verbose:
                sys.stdout.buffer.write(next_bytes)
                sys.stdout.buffer.flush()
            return logits

        def should_stop(tokens_so_far, logits):
            nonlocal done
            if np.argmax(logits) == self.model.token_eos():
                done = True
            return done or len(tokens_so_far) == self.model.n_ctx()

        self.model.reset()
        seen_tokens = []
        decompressed = bytearray()
        token_decoder = Decoder(base64_to_bits(compressed))
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()
        done = False
        while not done:
            start_idx = max(0, len(seen_tokens) - window_overlap)
            consume(
                self.model.generate(
                    tokens=[self.model.token_bos()] + seen_tokens[start_idx:],
                    temp=0.0,
                    logits_processor=process_logits,
                    stopping_criteria=should_stop,
                )
            )
        return decompressed


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="LLM-powered lossless compression tool"
    )
    parser.add_argument("model_path", help="path to model file")
    parser.add_argument(
        "-w",
        "--window-overlap",
        dest="overlap",
        default="0%",
        help="how much model context (as number of tokens or percentage of model context length) to maintain after filling the window. higher values increase compression ratio but decrease speed. must use same value for compression and decompression (default: 0%%)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        help="model context length (default: 0, which uses maximum supported by the model)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="number of model layers to offload to GPU (default: -1, which offloads all layers)",
    )
    parser.add_argument(
        "--use-mlock",
        default=False,
        action="store_true",
        help="use mlock to keep model in RAM (disabled by default)",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-c",
        "--compress",
        dest="string",
        nargs="*",
        help="compress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-d",
        "--decompress",
        dest="compressed",
        nargs="*",
        help="decompress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        default=False,
        action="store_true",
        help="show a prompt for interactive compression and decompression",
    )
    return parser


def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    compressor = LlamaZip(
        model_path=args.model_path,
        use_mlock=args.use_mlock,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )

    try:
        if args.overlap.endswith("%"):
            percent = float(args.overlap[:-1])
            if not (0 <= percent <= 100):
                parser.error("window overlap must be in the range [0%, 100%]")
            window_overlap = int(percent / 100 * (compressor.model.n_ctx() - 1))
        else:
            window_overlap = int(args.overlap)
            if window_overlap < 0:
                window_overlap += compressor.model.n_ctx()
            if not (0 <= window_overlap < compressor.model.n_ctx()):
                parser.error(
                    f"window overlap must be in the range [{-compressor.model.n_ctx()}, {compressor.model.n_ctx() - 1}]"
                )
    except ValueError:
        parser.error(
            "window overlap must be an integer (number of tokens) or a percentage (of the model's context length)"
        )

    try:
        if args.string is not None:
            uncompressed = (
                " ".join(args.string).encode("utf-8")
                if args.string
                else sys.stdin.buffer.read()
            )
            compressor.compress(uncompressed, window_overlap)
        elif args.compressed is not None:
            compressed = (
                args.compressed[0].encode("utf-8")
                if args.compressed
                else sys.stdin.buffer.read()
            )
            if not all(char in BASE64 for char in compressed):
                parser.error("compressed data must be base64-encoded")
            compressor.decompress(compressed, window_overlap)
        elif args.interactive:
            while True:
                try:
                    string = input("≥≥≥ ").encode("utf-8")
                except UnicodeDecodeError:
                    print(
                        "error: interactive mode only supports UTF-8 input",
                        file=sys.stderr,
                    )
                    continue
                if string and all(char in BASE64 for char in string):
                    try:
                        compressor.decompress(string, window_overlap)
                    except KeyboardInterrupt:
                        pass
                else:
                    compressor.compress(string, window_overlap)
                print("\n", file=sys.stderr)
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
