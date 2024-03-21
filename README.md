# langchain-snowpoc

> **NOTE**: This is just a PoC, not production-ready and covers just a few use cases.

This is a PoC of how can Cortex be used with Langchain.

It shows how easy it is to integrate Langchain and Cortex.

## Setup

```bash
conda env create
```

To use the environment you have to activate it. Feel free to check the [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html).

## Jupyter Lab Example

Just run `jupyter lab` in the main folder. Open `quickstart-jupyter.ipynb`

## Streamlit example

Just run the following code:

```bash
streamlit run quickstart-streamlit.py
```

Wait some time for document to be downloaded and chunked and ask questions regarding it.

For example:

* What is the document about?
* Who is mentioned in the document?
* When was the document created?
