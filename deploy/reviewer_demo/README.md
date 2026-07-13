# Minimal reviewer demo

`app.py` is the source embedded in the canonical Colab demo. It deliberately
exposes only four UI elements: History, Input, Send, and Reset.

The runtime downloads the public adapter without credentials, verifies both
adapter files by SHA-256, loads the pinned Qwen base in 4-bit, and supplies
only the learned fixed control token plus visible chat history.

The download is pinned to immutable Hugging Face model commit
`c20d382f32810deaff2f691cdf78c0a3a4d9be59`, the commit that published the
verified exact-silver v1 adapter files.

Use the generated notebook rather than running this file on a laptop:

`submission/notebooks/03_demo_uae_adab_tutor.ipynb`

The final cell uses Colab's session proxy rather than the external
`gradio.live` tunnel. It renders the chat inline and prints a clickable
`https://localhost:...` label that opens the proxied interface in a separate
tab. Keep the final cell running while using either view.

A CUDA GPU with at least T4-class memory is required.
