import argparse
import signal
import string
import sys
from collections import deque

import numpy as np
from llama_cpp import Llama
from more_itertools import consume
from tqdm import tqdm


NUM_STATE_BITS = 64
FREQ_SCALE_FACTOR = 1 << 32
BASE64 = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"


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
    def __init__(self, encoded_data: list):
        super().__init__()
        self.input = deque(encoded_data)
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


def compute_cdf(logits):
    logprobs = model.logits_to_logprobs(logits)
    probs = np.exp(logprobs)
    freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
    cum_freqs = np.cumsum(freqs)
    return cum_freqs


def compress(string, window_overlap):
    def bits_to_base64(bits):
        while bits and bits[-1] == 0:
            bits.pop()
        while len(bits) % 6 != 0:
            bits.append(0)
        return "".join(
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
            print(file=sys.stderr)
        next_token = tokens[next_token_idx]
        next_token_idx += 1
        cdf = compute_cdf(logits)
        encoder.encode_symbol(cdf, next_token)
        progress_bar.update()
        logits[next_token] = np.inf
        return logits

    def should_stop(tokens_so_far, logits):
        return (
            np.argmax(logits) == model.token_eos()
            or len(tokens_so_far) == model.n_ctx()
        )

    model.reset()
    tokens = model.tokenize(string.encode("utf-8"), add_bos=False)
    tokens.append(model.token_eos())
    next_token_idx = 0
    encoder = Encoder()

    interrupted = False
    s = signal.signal(signal.SIGINT, sigint_handler)

    progress_bar = tqdm(
        total=len(tokens),
        mininterval=1 / 30,
        desc="Compressing",
        unit="tok",
        leave=False,
        dynamic_ncols=True,
    )
    while next_token_idx < len(tokens):
        start_idx = max(0, next_token_idx - window_overlap)
        consume(
            model.generate(
                tokens=[model.token_bos()] + tokens[start_idx:next_token_idx],
                temp=0.0,
                logits_processor=process_logits,
                stopping_criteria=should_stop,
            )
        )
    progress_bar.close()

    encoder.finish()
    compressed = bits_to_base64(encoder.get_encoded())
    print(compressed)

    signal.signal(signal.SIGINT, s)

    return compressed


def decompress(compressed, window_overlap):
    def base64_to_bits(string):
        bits = [int(bit) for char in string for bit in f"{BASE64.index(char):06b}"]
        return bits

    def process_logits(_, logits):
        cdf = compute_cdf(logits)
        next_token = decoder.decode_symbol(cdf)
        logits[next_token] = np.inf
        if next_token == model.token_eos():
            return logits
        tokens.append(next_token)
        output_buf.extend(model.detokenize([next_token]))
        try:
            print(output_buf.decode("utf-8"), end="", flush=True)
            output_buf.clear()
        except UnicodeDecodeError:
            pass
        return logits

    def should_stop(tokens_so_far, logits):
        nonlocal done
        if np.argmax(logits) == model.token_eos():
            done = True
        return done or len(tokens_so_far) == model.n_ctx()

    model.reset()
    tokens = []
    encoded = base64_to_bits(compressed)
    decoder = Decoder(encoded)
    output_buf = bytearray()
    done = False
    while not done:
        overlap_start_idx = max(0, len(tokens) - window_overlap)
        consume(
            model.generate(
                tokens=[model.token_bos()] + tokens[overlap_start_idx:],
                temp=0.0,
                logits_processor=process_logits,
                stopping_criteria=should_stop,
            )
        )
    decompressed = model.detokenize(tokens).decode("utf-8")
    return decompressed


def load_model(model_path):
    global model
    loading_message = "Loading model..."
    print(loading_message, end="", flush=True, file=sys.stderr)
    model = Llama(
        model_path,
        n_gpu_layers=-1,
        verbose=False,
        use_mlock=True,
        n_ctx=0,
    )
    print("\r" + " " * len(loading_message) + "\r", end="", flush=True, file=sys.stderr)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered lossless compression tool"
    )

    parser.add_argument("model_path", help="path to a .gguf model file")

    parser.add_argument(
        "-w",
        "--window-overlap",
        dest="overlap",
        default="0%",
        help="how much context (as number of tokens or percent of window) to maintain after reaching the model's context limit (default: 0%%)",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-c",
        "--compress",
        dest="string",
        nargs="*",
        help="comppress input string or stdin",
    )
    mode_group.add_argument(
        "-d",
        "--decompress",
        dest="compressed",
        nargs="*",
        help="decompress input string or stdin",
    )
    mode_group.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        default=False,
        action="store_true",
        help="show a prompt for interactive compression and decompression",
    )

    args = parser.parse_args()

    load_model(args.model_path)

    try:
        if args.overlap.endswith("%"):
            percent = float(args.overlap[:-1])
            if not (0 <= percent <= 100):
                parser.error("window overlap must be in the range [0%, 100%]")
            window_overlap = int(percent / 100 * (model.n_ctx() - 1))
        else:
            window_overlap = int(args.overlap)
            if window_overlap < 0:
                window_overlap += model.n_ctx()
            if not (0 <= window_overlap < model.n_ctx()):
                parser.error(
                    f"window overlap must be in the range [{-model.n_ctx()}, {model.n_ctx() - 1}]"
                )
    except ValueError:
        parser.error(
            f"window overlap must be an integer (number of tokens) or a percentage (of the model's context length)"
        )

    try:
        if args.string is not None:
            string = " ".join(args.string) if args.string else sys.stdin.read()
            compress(string, window_overlap)
        elif args.compressed is not None:
            compressed = (
                args.compressed[0] if args.compressed else sys.stdin.read().strip()
            )
            if not all(char in BASE64 for char in compressed):
                parser.error("invalid compressed string")
            decompress(compressed, window_overlap)
        elif args.interactive:
            while True:
                print("≥≥≥", end=" ", flush=True, file=sys.stderr)
                string = input()
                if string and all(char in BASE64 for char in string):
                    try:
                        decompress(string, window_overlap)
                    except KeyboardInterrupt:
                        pass
                    print(file=sys.stderr)
                else:
                    compress(string, window_overlap)
                print(file=sys.stderr)
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
