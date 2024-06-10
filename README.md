# llama-zip

`llama-zip` is a command-line utility for lossless text compression and decompression. It functions by leveraging a user-provided LLM (large language model) as the probabilistic model for an [arithmetic coder](https://en.wikipedia.org/wiki/Arithmetic_coding). This allows `llama-zip` to achieve high compression ratios for structured or natural language text, as fewer bits are needed to encode tokens that the LLM predicts with high confidence. By employing a sliding context window, `llama-zip` is not limited by the LLM's maximum context length and can handle arbitrarily long input text. The main limitation of `llama-zip` is that the speed of compression and decompression is limited by the LLM's inference speed.

![Interactive Mode Demo: Lorem Ipsum Text](lorem_ipsum_demo.gif)

## Compression Performance

In the table below, the compression ratios achieved by `llama-zip` on the text files of the [Calgary Corpus](http://www.data-compression.info/Corpora/CalgaryCorpus/) (as well as on `llama-zip`'s own source code) are compared to other popular or high-performance compression utilities. Compression ratios are calculated by dividing the number of bytes in the uncompressed input by the number of bytes in the compressed output, so higher values indicate more effective compression. For `llama-zip`, the LLM used was [Llama 3 8B (Q4_K_M)](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF) with a window overlap of 25%. For the other utilities, the maximum compression level offered was used.

| File         |        llama-zip |          paq8pxd |             zpaq | bzip2 |  lzma |    xz |  zstd |  gzip |
| :----------- | ---------------: | ---------------: | ---------------: | ----: | ----: | ----: | ----: | ----: |
| bib          |        **8.523** | <ins>5.590</ins> |            4.611 | 4.051 | 3.641 | 3.636 | 3.485 | 3.171 |
| book1        |        **6.943** | <ins>4.204</ins> |            3.823 | 3.305 | 2.942 | 2.941 | 2.904 | 2.460 |
| book2        |        **8.127** | <ins>5.325</ins> |            4.649 | 3.880 | 3.598 | 3.596 | 3.514 | 2.963 |
| news         |        **5.590** | <ins>4.494</ins> |            3.817 | 3.180 | 3.173 | 3.171 | 3.073 | 2.610 |
| paper1       |        **7.637** | <ins>4.212</ins> |            3.572 | 3.211 | 3.083 | 3.074 | 3.017 | 2.867 |
| paper2       |        **8.375** | <ins>4.135</ins> |            3.679 | 3.283 | 3.020 | 3.015 | 2.982 | 2.769 |
| progc        |        **4.425** | <ins>4.352</ins> |            3.495 | 3.158 | 3.162 | 3.151 | 3.096 | 2.968 |
| progl        |            5.194 |        **7.347** | <ins>5.554</ins> | 4.599 | 4.801 | 4.787 | 4.728 | 4.432 |
| progp        | <ins>6.309</ins> |        **7.508** |            5.348 | 4.611 | 4.792 | 4.772 | 4.724 | 4.414 |
| trans        |        **9.810** | <ins>8.409</ins> |            6.597 | 5.235 | 5.628 | 5.613 | 5.417 | 4.949 |
| llama_zip.py |        **5.859** | <ins>4.689</ins> |            3.018 | 3.508 | 3.608 | 3.552 | 3.633 | 3.542 |

The best-performing compressor for each file is listed in bold, and the second-best is underlined. The columns are sorted by average compression ratio achieved across all files, with overall better-performing compressors listed further to the left.

## Setup

```sh
git clone https://github.com/alexbuz/llama-zip.git
cd llama-zip
pip3 install .
```

### LLM Download

To use `llama-zip`, you must first download an LLM that is compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp), such as [Llama 3 8B](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF). Make sure to download a quantized version (one of the `.gguf` files listed on the "Files and versions" tab on Hugging Face) that is small enough to fit in your system's memory.

## Usage

```
llama-zip <llm_path> [options] <mode> [input]
```

### Modes

`llama-zip` supports three modes of operation:

1. **Compress mode** (specified by the `-c` or `--compress` flag): The string to be compressed can be provided as an argument or piped to stdin. The compressed output will be encoded in base64 and printed to stdout.
2. **Decompress mode** (specified by the `-d` or `--decompress` flag): The compressed string can be provided as an argument or piped to stdin. The decompressed output will be printed to stdout.
3. **Interactive mode** (specified by the `-i` or `--interactive` flag): A prompt is displayed where the user can enter strings to be compressed or decompressed. When a base64-encoded string is entered, it will be decompressed; otherwise, the entered string will be compressed. After each compression or decompression operation, the user is prompted to enter another string. To exit interactive mode, press `Ctrl+C`.
    - **Note:** If you would like to compress a string that consists entirely of base64 characters (i.e., letters, numbers, `+`, and `/`, without any other symbols or spaces), you must use compression mode directly, as interactive mode assumes that base64-encoded strings are meant to be decompressed and will result in nonsensical output if the input did not come from a compression operation. Alternatively, you can add a non-base64 character to your string (such as a space at the end) if you don't mind your string being compressed with that extra character.

### Options

- `-w`, `--window-overlap`: The number of tokens to overlap between the end of the previous context window and the start of the next window, when compressing a string whose length exceeds the LLM's maximum context length. This can be specified as a percentage of the LLM's context length or as a fixed number of tokens. The default is `0%`, meaning that the context window is cleared entirely when it is filled. Higher values can improve compression ratios but will slow down compression and decompression, since parts of the text will need to be re-evaluated when the context window slides. Note that when decompressing, the window overlap must be set to the same value that was used during compression in order to recover the original text.
- `--n_gpu_layers`: The `--n_gpu_layers` argument in the code specifies the number of layers in the model that should be offloaded to the GPU for computation. This can significantly speed up the processing time, especially for larger models, as the GPU is typically much faster at performing matrix operations than a CPU. If `--n_gpu_layers` is set to -1 or None, all layers of the model will be offloaded to the GPU. Check [llama.cpp's](https://github.com/ggerganov/llama.cpp) readme for better understanding of this parameter.

### Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKL-ZVucgbVcZnEi9NyfjMIJ_PrTKMEW?usp=sharing)

#### Compression
- Compressing a string:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -c "The quick brown fox jumps over the lazy dog."
    # Output: SxapgbY
    ```

- Compressing text from a file:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -c < /path/to/gettysburg_address.txt
    # Output: 4vTMmKKTXWAcNZwPwkqN84
    ```

- Compressing text from a file and saving the output to another file:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -c < /path/to/input.txt > /path/to/output.compressed
    ```

#### Decompression
- Decompressing a compressed string:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -d SxapgbY
    # Output: The quick brown fox jumps over the lazy dog.
    ```

- Decompressing text from a file:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -d < /path/to/input.compressed
    # Output: [decompressed text]
    ```

- Decompressing text from a file and saving the output to another file:
    ```sh
    llama-zip /path/to/Meta-Llama-3-8B.Q8_0.gguf -d < /path/to/input.compressed > /path/to/output.txt
    ```
