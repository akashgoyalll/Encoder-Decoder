# Encoder–Decoder NLP

A [Streamlit](https://streamlit.io/) web app that runs two **sequence-to-sequence (seq2seq)** models built with **TensorFlow/Keras**: **English → Hindi translation** and **English text summarization**. Both use teacher-forcing–trained models with separate **inference encoder/decoder** graphs (greedy decoding; translation can fall back to beam search when output looks degenerate; summarization can fall back to an extractive summary when the neural output is weak).

## Features

- **Translator mode** — Encode English, decode Hindi with greedy decoding; beam search as a rescue when greedy output looks collapsed.
- **Summarizer mode** — Encode a document, decode a short summary; optional extractive fallback (sentence scoring by word frequency) when overlap with the source is too low.

<img width="1917" height="905" alt="Screenshot 2026-04-22 001642" src="https://github.com/user-attachments/assets/62b49afa-f877-40d0-a84c-9012378af42e" />

-

<img width="1915" height="912" alt="image" src="https://github.com/user-attachments/assets/3e474dd7-fe09-47b9-92ba-96013b178bee" />



## Requirements

- Python 3.10+ recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually: `tensorflow`, `streamlit`, `numpy` (see `requirements.txt`).

## Model files (required)

Place these next to `app.py` (same folder as the script):

| File | Purpose |
|------|---------|
| `eng_hin_translation_model.h5` | English→Hindi seq2seq weights |
| `tokenizer_data.pkl` | English/Hindi tokenizers and lengths (`eng_tokenizer`, `hin_tokenizer`, `max_eng_len`, `max_hin_len`, `latent_dim`) |
| `custom_summarizer_model.keras` | Summarizer seq2seq weights |
| `summarizer_tokenizer_data.pkl` | Text/summary tokenizers and lengths |

Without these files, the app will fail when you click **Generate**.

## Run the app

From the project directory:

```bash
streamlit run app.py
```

On Windows, you can run `.\run_streamlit.ps1` from PowerShell (uses `python` on your `PATH`; activate a virtual environment first if you use one).

The UI opens in your browser (default Streamlit port **8501**).

## Usage

1. Choose **Translator Mode** or **Summarizer Mode** in the sidebar.
2. Paste or type your text.
3. Click **Generate**.

**Notes**

- Vocabulary is limited (e.g. top ~10k English words for translation, ~8k for summarization). Rare words become `<unk>` and can reduce quality—prefer common words when testing.
- Long inputs are padded/truncated to the max lengths stored in the pickle files.

## Example sentences

Use these to try the app after models are in place.

### Translation (English → Hindi)

Short, everyday lines:

| English |
|--------|
| Hello, how are you today? |
| The weather is nice this morning. |
| I would like a cup of tea, please. |
| Where is the nearest train station? |
| She reads books every evening. |
| We are learning machine learning this semester. |

Slightly longer:

| English |
|--------|
| The children played in the park until the sun went down. |
| If you need help, please call me tomorrow morning. |

### Summarization (English document → short summary)

Paste a short paragraph like:

**Example A — news-style**

> Technology companies announced new policies for remote work this year. Employees can choose hybrid schedules with at least two days in the office. The changes aim to improve collaboration while keeping flexibility for families. Critics argue that commuting costs remain high in major cities.

**Example B — simple story**

> Maria opened the small bakery on Maple Street five years ago. She bakes bread before sunrise and serves coffee to regular customers. Last month she hired an assistant to help with weekend crowds. The shop plans to add outdoor seating next spring.

**Example C — factual blurb**

> Water boils at 100 degrees Celsius at sea level. At higher altitude, the boiling point is lower because atmospheric pressure drops. This is why cooking times for pasta may need adjustment in mountain regions.

You should see a condensed summary in **Summarizer Mode**; if the model’s output is too generic, the app may switch to an extractive summary automatically.

## Project layout

```text
EncoderDecoder/
├── app.py                 # Streamlit UI + inference (encoder/decoder, beam, fallbacks)
├── requirements.txt       # Python dependencies
├── run_streamlit.ps1      # Optional Windows launcher (python on PATH)
└── README.md
```

Trained model artifacts (`.h5`, `.keras`, `.pkl`) are expected beside `app.py` but are usually not committed to Git due to size—add them locally or via your own release assets.

