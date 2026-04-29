# Contributing

Contributions are welcome — bug fixes, new exercises, better visualizations, or additional neuroscience models.

## How to Contribute

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Test notebooks on Colab (Runtime → Run all)
5. Submit a pull request with a clear description

## Notebook Standards

- All notebooks must run top-to-bottom on Google Colab with a T4 GPU runtime
- Include `!nvidia-smi` at the top of every CUDA notebook
- Clear all outputs before committing (`Cell → All Output → Clear`)
- Markdown cells should explain the *why*, not just the *what*

## CUDA Code Standards

- Use `cudaCheckError()` macro for all CUDA API calls
- Comment non-obvious kernel logic
- Provide timing with `cudaEvent` where performance is discussed

## Reporting Issues

Open a GitHub Issue with:
- Which notebook/module
- Colab runtime info (`!nvidia-smi` output)
- Error message (full traceback)
