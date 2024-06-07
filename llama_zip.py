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


def compress(string):
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
        if interrupted and len(tokens) > 1:
            tokens.clear()
            tokens.append(model.token_eos())
            print(file=sys.stderr)
        next_token = tokens.popleft()
        cdf = compute_cdf(logits)
        encoder.encode_symbol(cdf, next_token)
        logits[next_token] = np.inf
        return logits

    def should_stop(_, logits):
        return np.argmax(logits) == model.token_eos()

    model.reset()
    tokens = deque(model.tokenize(string.encode("utf-8"), add_bos=False))
    if len(tokens) >= model.n_ctx():
        print(
            f"Error: Input length ({len(tokens)} tokens) exceeds maximum for model ({model.n_ctx() - 1} tokens).",
            file=sys.stderr,
        )
        exit(1)
    tokens.append(model.token_eos())
    encoder = Encoder()

    interrupted = False
    s = signal.signal(signal.SIGINT, sigint_handler)

    consume(
        tqdm(
            model.generate(
                tokens=[model.token_bos()],
                temp=0.0,
                logits_processor=process_logits,
                stopping_criteria=should_stop,
            ),
            total=len(tokens) - 1,
            mininterval=1 / 30,
            desc="Compressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
        )
    )

    encoder.finish()
    compressed = bits_to_base64(encoder.get_encoded())
    print(compressed)

    signal.signal(signal.SIGINT, s)

    return compressed


def decompress(compressed):
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

    def should_stop(_, logits):
        return np.argmax(logits) == model.token_eos()

    model.reset()
    tokens = []
    encoded = base64_to_bits(compressed)
    decoder = Decoder(encoded)
    output_buf = bytearray()
    consume(
        model.generate(
            tokens=[model.token_bos()],
            temp=0.0,
            logits_processor=process_logits,
            stopping_criteria=should_stop,
        )
    )
    decompressed = model.detokenize(tokens).decode("utf-8")
    return decompressed


def print_usage_and_exit():
    print(
        f"""\
Usage: {sys.argv[0]} <llm_path> <mode>
Modes:
  -c, --compress [string]
  -d, --decompress [compressed_string]
  -i, --interactive
For compression and decompression, the input is read from stdin if not provided as an argument.""",
        file=sys.stderr,
    )
    exit(1)


def load_model(model_path):
    global model
    loading_message = "Loading model..."
    print(loading_message, end="", flush=True, file=sys.stderr)
    model = Llama(
        model_path,
        n_gpu_layers=-1,
        verbose=False,
        use_mlock=True,
        logits_all=True,
        n_ctx=0,
    )
    print("\r" + " " * len(loading_message) + "\r", end="", flush=True, file=sys.stderr)
    return model


def main():
    try:
        global model_path
        model_path = sys.argv[1]
    except IndexError:
        print_usage_and_exit()

    try:
        mode = sys.argv[2]
    except IndexError:
        print_usage_and_exit()

    try:
        if mode in ["-c", "--compress"]:
            try:
                string = " ".join([sys.argv[3], *sys.argv[4:]])
            except IndexError:
                string = sys.stdin.read()
            load_model(model_path)
            compress(string)
        elif mode in ["-d", "--decompress"]:
            try:
                compressed = sys.argv[3]
            except IndexError:
                compressed = sys.stdin.read().strip()
            load_model(model_path)
            decompress(compressed)
        elif mode in ["-i", "--interactive"]:
            load_model(model_path)
            while True:
                print("≥≥≥", end=" ", flush=True, file=sys.stderr)
                string = input()
                if string and all(char in BASE64 for char in string):
                    try:
                        decompress(string)
                    except KeyboardInterrupt:
                        pass
                    print(file=sys.stderr)
                else:
                    compress(string)
                print(file=sys.stderr)
        else:
            print_usage_and_exit()
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
