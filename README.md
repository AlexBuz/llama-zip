# llama-zip

llama-zip is a lossless compression utility that leverages a user-provided LLM (large language model) as the probabilistic model for an [arithmetic coder](https://en.wikipedia.org/wiki/Arithmetic_coding). This allows llama-zip to achieve high compression ratios on structured or natural language text, since few bits are needed to encode tokens that the model predicts with high confidence. By employing a sliding context window, llama-zip is not limited by the context length of the LLM and can compress strings of arbitrary length. Furthermore, by encoding invalid UTF-8 bytes using code points in Unicode's private use areas, llama-zip is not limited to text inputs and can handle arbitrary binary data, albeit with reduced compression ratios compared to text inputs.

![Interactive Mode Demo: Lorem Ipsum Text](lorem_ipsum_demo.gif)

## Compression Performance

In the table below, the compression ratios achieved by llama-zip on the text files of the [Calgary Corpus](http://www.data-compression.info/Corpora/CalgaryCorpus/) (as well as on llama-zip's own source code, `llama_zip.py`) are compared to other popular or high-performance compression utilities. Compression ratios are calculated by dividing the number of bytes in the uncompressed input by the number of bytes in the compressed output, so higher values indicate more effective compression.

For llama-zip, two different LLMs were benchmarked, with varying context lengths but with a consistent window overlap of 25% (see the [Options](#options) section below for information about these parameters). The models used were as follows:
- [Phi-3 Mini-128K-Instruct (Q4_K_M)](https://huggingface.co/QuantFactory/Phi-3-mini-128k-instruct-GGUF)
  - "Phi&#8209;32k" - 32768-token context length
  - "Phi&#8209;8k" - 8192-token context length
- [Llama 3 8B (Q4_K_M)](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF)
  - "Llama-8k" - 8192-token context length (the maximum for this model)

For the other utilities, the maximum compression level offered was used.

| File         | llama&#8209;zip (Phi&#8209;32k) | llama&#8209;zip (Phi&#8209;8k) | llama&#8209;zip (Llama&#8209;8k) |  cmix | paq8px | paq8pxd |  zpaq | brotli | bzip2 |  lzma |    xz |  zstd |  gzip |
| :----------- | ------------------------------: | -----------------------------: | -------------------------------: | ----: | -----: | ------: | ----: | -----: | ----: | ----: | ----: | ----: | ----: |
| bib          |                       **9.845** |               <ins>9.286</ins> |                            8.523 | 5.633 |  5.668 |   5.590 | 4.611 |  3.920 | 4.051 | 3.641 | 3.636 | 3.485 | 3.171 |
| book1        |                <ins>6.876</ins> |                          6.872 |                        **6.943** | 4.209 |  4.192 |   4.204 | 3.823 |  2.999 | 3.305 | 2.942 | 2.941 | 2.904 | 2.460 |
| book2        |                      **10.213** |               <ins>9.716</ins> |                            8.127 | 5.381 |  5.346 |   5.325 | 4.649 |  3.696 | 3.880 | 3.598 | 3.596 | 3.514 | 2.963 |
| news         |                       **7.965** |               <ins>7.566</ins> |                            5.590 | 4.542 |  4.531 |   4.494 | 3.817 |  3.338 | 3.180 | 3.173 | 3.171 | 3.073 | 2.610 |
| paper1       |                       **9.911** |               <ins>9.573</ins> |                            7.637 | 4.264 |  4.302 |   4.212 | 3.572 |  3.439 | 3.211 | 3.083 | 3.074 | 3.017 | 2.867 |
| paper2       |                      **10.379** |              <ins>10.156</ins> |                            8.375 | 4.180 |  4.208 |   4.135 | 3.679 |  3.308 | 3.283 | 3.020 | 3.015 | 2.982 | 2.769 |
| progc        |                      **10.120** |               <ins>9.680</ins> |                            4.425 | 4.439 |  4.438 |   4.352 | 3.495 |  3.409 | 3.158 | 3.162 | 3.151 | 3.096 | 2.968 |
| progl        |                      **13.662** |              <ins>13.086</ins> |                            5.194 | 7.497 |  7.464 |   7.347 | 5.554 |  5.116 | 4.599 | 4.801 | 4.787 | 4.728 | 4.432 |
| progp        |                      **15.378** |              <ins>14.238</ins> |                            6.309 | 7.705 |  7.665 |   7.508 | 5.348 |  4.998 | 4.611 | 4.792 | 4.772 | 4.724 | 4.414 |
| trans        |                      **10.348** |                          8.805 |                 <ins>9.810</ins> | 8.650 |  8.484 |   8.409 | 6.597 |  6.083 | 5.235 | 5.628 | 5.613 | 5.417 | 4.949 |
| llama_zip.py |                      **22.627** |                     **22.627** |                 <ins>5.849</ins> | 4.904 |  4.976 |   4.689 | 3.018 |  3.980 | 3.508 | 3.608 | 3.552 | 3.633 | 3.542 |

The best-performing compressor for each file is listed in bold, and the second-best is underlined. The columns are sorted by average compression ratio achieved across all files, with overall better-performing compressors listed further to the left.

## Installation

```sh
git clone https://github.com/alexbuz/llama-zip.git
cd llama-zip
pip3 install .
```

### LLM Download

To use llama-zip, you must download an LLM that is compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp), such as [Llama 3 8B](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF). Make sure to download a quantized version (one of the `.gguf` files listed on the "Files and versions" tab on Hugging Face) that is small enough to fit in your system's memory.

## CLI Usage

```
llama-zip <llm_path> [options] <mode> [input]
```

### Modes

llama-zip supports three modes of operation:

1. **Compress mode** (specified by the `-c` or `--compress` flag): The string to be compressed can be provided as an argument or piped to stdin. The compressed output will be written to stdout.
2. **Decompress mode** (specified by the `-d` or `--decompress` flag): The compressed string can be provided as an argument or piped to stdin. The decompressed output will be written to stdout.
3. **Interactive mode** (specified by the `-i` or `--interactive` flag): A prompt is displayed where the user can enter strings to be compressed or decompressed. When a base64-encoded string is entered, it will be treated as representing compressed data and will be decompressed; otherwise, it will be compressed. After each compression or decompression operation, the user is prompted to enter another string. To exit interactive mode, press `Ctrl+C`.
    - **Note:** If you would like to compress a string that consists entirely of base64 characters (i.e., letters, numbers, `+`, and `/`, without any other symbols or spaces), you must use compression mode directly, as interactive mode assumes that base64-encoded strings are meant to be decompressed and will result in nonsensical output if the input did not come from a compression operation. Alternatively, you can add a non-base64 character to your string (such as a space at the end) if you don't mind that character being compressed along with the rest of the string.

### Options

- `-f`, `--compressed-format`: The format of compressed data. This can be set to `binary` (the default for non-interactive modes) or `base64` (the default and only supported format for interactive mode).
- `-w`, `--window-overlap`: The number of tokens to overlap between the end of the previous context window and the start of the next window, when compressing a string whose length exceeds the model's maximum context length. This can be specified as a percentage of the model's context length or as a fixed number of tokens. The default is `0%`, meaning that the context window is cleared entirely when it is filled. Higher values can improve compression ratios but will slow down compression and decompression. Note that when decompressing, the window overlap must be set to the same value that was used during compression in order to reconstruct the original string.
- `--n-ctx`: The number of tokens to use as the context length for the model. This must be less than or equal to the model's maximum context length. If set to `0` (the default), then the model's maximum context length will be used. Note that when decompressing, the context length must be set to the same value that was used during compression in order to reconstruct the original string.
- `--n-gpu-layers`: The number of model layers to offload to the GPU. This can significantly speed up compression and decompression, especially for larger models. If set to `-1` (the default), then all layers will be offloaded. See the [llama.cpp repository](https://github.com/ggerganov/llama.cpp) for more information. In [practice](#limitations), the same number of layers should be offloaded during compression and decompression.
- `--use-mlock`: Force your system to keep the entire model in memory. This can be useful for larger models but may cause your system to run out of memory if the model is too large. Disabled by default.

### Examples

#### Compression

- Compress a file:
    ```sh
    llama-zip /path/to/llm.gguf -c < input.txt > compressed.llzp
    ```

- Compress a string and print the compressed output in base64 format:
    ```sh
    llama-zip /path/to/llm.gguf -f base64 -c "The quick brown fox jumps over the lazy dog."
    ```

#### Decompression

- Decompress a file:
    ```sh
    llama-zip /path/to/llm.gguf -d < compressed.llzp > output.txt
    ```

- Decompress a base64-encoded compressed string:
    ```sh
    llama-zip /path/to/llm.gguf -f base64 -d BASE64_STRING
    ```

#### Interactive Mode

- Start an interactive mode session:
    ```sh
    llama-zip /path/to/llm.gguf -i
    ```

#### Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKL-ZVucgbVcZnEi9NyfjMIJ_PrTKMEW?usp=sharing)

## API Usage

The `LlamaZip` class can be used to compress and decompress data programmatically. The `compress` method takes a `bytes` object and returns a another `bytes` object containing the compressed data. The `decompress` method takes a `bytes` object containing compressed data and returns the original uncompressed data.

```python
from llama_zip import LlamaZip

# Initialize the compressor
compressor = LlamaZip(model_path="/path/to/model.gguf")

# Compress some data
original = b"The quick brown fox jumps over the lazy dog."
compressed = compressor.compress(original)
assert len(compressed) < len(original)

# Decompress the data
decompressed = compressor.decompress(compressed)
assert decompressed == original
```

The `LlamaZip` constructor also accepts the `n_ctx`, `n_gpu_layers`, and `use_mlock` arguments, which correspond to the CLI options of the same names. The `window_overlap` argument can be passed to the `compress` and `decompress` methods directly to specify the window overlap for that particular operation.

## Limitations

1. **Speed:** Compression and decompression speeds are limited by the speed of LLM inference. This renders llama-zip significantly slower than traditional compression utilities. However, the compression ratios achieved by llama-zip may justify the trade-off in speed for certain use cases.
2. **Portability:** llama-zip requires identical LLM behavior during compression and decompression. However, the backend that llama-zip uses for LLM inference, [llama.cpp](https://github.com/ggerganov/llama.cpp), does not currently guarantee deterministic behavior. This limits the portability of the compressed output of llama-zip, as it may not be decompressible on a different system, even if the same model is used. In practice, behavior also differs depending on the number of GPU layers offloaded, so the `--n-gpu-layers` option should be set to the same value during compression and decompression, in addition to the window overlap (`--window-overlap`) and context length (`--n-ctx`) options.
3. **Binary Compression:** Due to its reliance on an LLM for prediction, llama-zip is best suited for compressing inputs that consist primarily of text. Although llama-zip can handle binary data by encoding invalid UTF-8 bytes using code points in Unicode's private use areas, it may not achieve high compression ratios on such data, potentially producing compressed output that is larger than the original input.