"""
Streamlit app: English→Hindi translation and text summarization using
teacher-forcing seq2seq Keras models with proper inference encoder/decoder graphs.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# Paths (same directory as this script)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRANSLATION_MODEL_PATH = BASE_DIR / "eng_hin_translation_model.h5"
TRANSLATION_TOKENIZER_PATH = BASE_DIR / "tokenizer_data.pkl"
SUMMARIZER_MODEL_PATH = BASE_DIR / "custom_summarizer_model.keras"
SUMMARIZER_TOKENIZER_PATH = BASE_DIR / "summarizer_tokenizer_data.pkl"

# Layer names match the saved Functional models (see model configs).
TRANSLATION_LAYER_NAMES = {
    "enc_emb": "embedding_2",
    "enc_lstm": "lstm_2",
    "dec_emb": "embedding_3",
    "dec_lstm": "lstm_3",
    "dense": "dense_1",
}

SUMMARIZER_LAYER_NAMES = {
    "enc_emb": "embedding",
    "enc_lstm": "lstm",
    "dec_emb": "embedding_1",
    "dec_lstm": "lstm_1",
    "dense": "dense",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


# -----------------------------------------------------------------------------
# Pickle + model loading (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_translation_assets():
    """Load translation model, tokenizers, and build inference encoder/decoder."""
    model = load_model(str(TRANSLATION_MODEL_PATH), compile=False)
    with open(TRANSLATION_TOKENIZER_PATH, "rb") as f:
        data = pickle.load(f)

    eng_tok = data["eng_tokenizer"]
    hin_tok = data["hin_tokenizer"]
    max_eng_len = int(data["max_eng_len"])
    max_hin_len = int(data["max_hin_len"])
    latent_dim = int(data["latent_dim"])

    enc_model, dec_model = build_inference_models(
        model, TRANSLATION_LAYER_NAMES, latent_dim
    )
    return {
        "full_model": model,
        "encoder_model": enc_model,
        "decoder_model": dec_model,
        "eng_tokenizer": eng_tok,
        "hin_tokenizer": hin_tok,
        "max_src_len": max_eng_len,
        "max_tgt_len": max_hin_len,
        "latent_dim": latent_dim,
    }


@st.cache_resource
def load_summarizer_assets():
    """Load summarizer model, tokenizers, and build inference encoder/decoder."""
    model = load_model(str(SUMMARIZER_MODEL_PATH), compile=False)
    with open(SUMMARIZER_TOKENIZER_PATH, "rb") as f:
        data = pickle.load(f)

    text_tok = data["text_tokenizer"]
    summary_tok = data["summary_tokenizer"]
    max_text_len = int(data["max_text_len"])
    max_summary_len = int(data["max_summary_len"])
    latent_dim = int(data["latent_dim"])

    enc_model, dec_model = build_inference_models(
        model, SUMMARIZER_LAYER_NAMES, latent_dim
    )
    return {
        "full_model": model,
        "encoder_model": enc_model,
        "decoder_model": dec_model,
        "text_tokenizer": text_tok,
        "summary_tokenizer": summary_tok,
        "max_src_len": max_text_len,
        "max_tgt_len": max_summary_len,
        "latent_dim": latent_dim,
    }


def build_inference_models(full_model: Model, names: dict, latent_dim: int):
    """
    From a teacher-forcing seq2seq Functional model, build:
    - Encoder: source sequence -> (state_h, state_c)
    - Decoder: one target token + states -> (softmax over vocab, new_h, new_c)

    Reuses trained Embedding / LSTM / Dense weights from `full_model`.
    """
    enc_emb = full_model.get_layer(names["enc_emb"])
    enc_lstm = full_model.get_layer(names["enc_lstm"])
    dec_emb = full_model.get_layer(names["dec_emb"])
    dec_lstm = full_model.get_layer(names["dec_lstm"])
    dec_dense = full_model.get_layer(names["dense"])

    # Original model has two inputs: encoder_inputs, decoder_inputs
    encoder_inputs = full_model.input[0]
    enc_emb_out = enc_emb(encoder_inputs)
    enc_lstm_out = enc_lstm(enc_emb_out, training=False)

    # return_state=True -> (last_output, h, c)
    _, enc_state_h, enc_state_c = unpack_lstm_outputs(enc_lstm_out)
    encoder_model = Model(
        inputs=encoder_inputs,
        outputs=[enc_state_h, enc_state_c],
        name="inference_encoder",
    )

    # One decoder timestep at inference time
    decoder_token_in = Input(shape=(1,), dtype="int32", name="decoder_token_in")
    decoder_state_h = Input(shape=(latent_dim,), name="decoder_state_h")
    decoder_state_c = Input(shape=(latent_dim,), name="decoder_state_c")

    dec_emb_out = dec_emb(decoder_token_in)
    dec_lstm_out = dec_lstm(
        dec_emb_out,
        initial_state=[decoder_state_h, decoder_state_c],
        training=False,
    )
    dec_seq, dec_h, dec_c = unpack_lstm_outputs(dec_lstm_out)
    decoder_probs = dec_dense(dec_seq)

    decoder_model = Model(
        inputs=[decoder_token_in, decoder_state_h, decoder_state_c],
        outputs=[decoder_probs, dec_h, dec_c],
        name="inference_decoder",
    )

    return encoder_model, decoder_model


def unpack_lstm_outputs(lstm_out):
    """Normalize Keras LSTM(return_state=True) return value to (seq, h, c)."""
    if isinstance(lstm_out, (list, tuple)):
        if len(lstm_out) == 3:
            return lstm_out[0], lstm_out[1], lstm_out[2]
    raise ValueError("Expected LSTM outputs (sequences, state_h, state_c).")


def idx_to_word(tokenizer, idx: int) -> str:
    """Map a tokenizer index to a word string."""
    if idx <= 0:
        return ""
    iw = getattr(tokenizer, "index_word", None)
    if isinstance(iw, dict) and idx in iw:
        return iw[idx]
    for word, i in tokenizer.word_index.items():
        if i == idx:
            return word
    return ""


def get_special_ids(tokenizer):
    """Resolve <start> / <end> indices from the target tokenizer."""
    wi = tokenizer.word_index
    start_id = wi.get("<start>")
    end_id = wi.get("<end>")
    if start_id is None:
        raise KeyError(
            "Target tokenizer must contain '<start>' in word_index for inference."
        )
    return start_id, end_id


def greedy_next_token_id(probs_step: np.ndarray, target_tokenizer) -> int:
    """
    Pick the next decoder token from a softmax slice.

    These models often assign the highest mass to `<unk>` (especially when the
    encoder saw many OOV tokens, or when the summarizer's head favors `<unk>` at
    step 1). Masking index 0 (padded target class) and `<unk>` before argmax
    matches the next most likely *real* token and avoids endless `<unk>` output.
    """
    p = np.asarray(probs_step, dtype=np.float64).reshape(-1).copy()
    if p.size == 0:
        return 0
    p[0] = -np.inf
    unk_id = target_tokenizer.word_index.get("<unk>")
    if unk_id is not None and unk_id < p.size:
        p[unk_id] = -np.inf
    return int(np.argmax(p))


def encode_and_pad(text: str, tokenizer, max_len: int) -> np.ndarray:
    """Tokenize a single string and pad (post) to max_len."""
    seq = tokenizer.texts_to_sequences([text.strip()])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    return padded.astype(np.int32)


def normalize_text(text: str) -> str:
    """Normalize free-form text before tokenization/scoring."""
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for extractive fallback summaries."""
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> list[str]:
    """Word tokenizer used by fallback scoring and overlap checks."""
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def extractive_fallback_summary(text: str, max_sentences: int = 2) -> str:
    """
    Build an extractive summary when neural output is generic/collapsed.
    Scores sentences by weighted word frequency.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    words = tokenize_words(text)
    if not words:
        return " ".join(sentences[:max_sentences]).strip()

    freq = Counter(words)
    scored = []
    for i, sent in enumerate(sentences):
        sent_words = tokenize_words(sent)
        if not sent_words:
            continue
        score = sum(freq[w] for w in sent_words) / max(len(sent_words), 1)
        scored.append((score, i, sent))

    if not scored:
        return " ".join(sentences[:max_sentences]).strip()

    # Keep highest scoring sentences, then restore original order.
    selected = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda t: t[1])
    return " ".join(s for _, _, s in selected).strip()


def should_use_fallback_summary(source_text: str, generated_summary: str) -> bool:
    """
    Detect low-confidence neural summaries (empty, very short, or low overlap).
    Low lexical overlap often signals decoder collapse into a generic template.
    """
    gen_words = [w for w in tokenize_words(generated_summary) if w not in STOPWORDS]
    src_words = {w for w in tokenize_words(source_text) if w not in STOPWORDS}

    if len(gen_words) < 4:
        return True
    if not src_words:
        return True

    overlap = sum(1 for w in gen_words if w in src_words) / len(gen_words)
    return overlap < 0.2


def decode_sequence(
    input_seq: np.ndarray,
    encoder_model: Model,
    decoder_model: Model,
    target_tokenizer,
    max_tgt_len: int,
) -> str:
    """
    Run encoder once, then greedy decode with the inference decoder one step at a time.
    Stops on <end> or max_tgt_len. Output text excludes <start> and <end>.
    """
    start_id, end_id = get_special_ids(target_tokenizer)

    state_h, state_c = encoder_model(input_seq, training=False)
    # Ensure numpy for the loop (works with tf.Tensor or np.ndarray)
    state_h = np.asarray(state_h)
    state_c = np.asarray(state_c)

    target_seq = np.array([[start_id]], dtype=np.int32)
    collected: list[str] = []

    for _ in range(max_tgt_len):
        dec_probs, state_h, state_c = decoder_model(
            [target_seq, state_h, state_c], training=False
        )
        dec_probs = np.asarray(dec_probs)
        state_h = np.asarray(state_h)
        state_c = np.asarray(state_c)

        # Shape: (batch, time, vocab) with time == 1
        next_id = greedy_next_token_id(dec_probs[0, -1, :], target_tokenizer)

        if end_id is not None and next_id == end_id:
            break
        if next_id == 0:
            break

        word = idx_to_word(target_tokenizer, next_id)
        if not word or word == "<unk>":
            target_seq = np.array([[next_id]], dtype=np.int32)
            continue
        if word in ("<start>", "<end>"):
            if word == "<end>":
                break
            target_seq = np.array([[next_id]], dtype=np.int32)
            continue

        collected.append(word)
        target_seq = np.array([[next_id]], dtype=np.int32)

    cleaned = [
        w for w in collected if w and w not in ("<start>", "<end>", "<unk>")
    ]
    return " ".join(cleaned).strip()


def decode_sequence_beam(
    input_seq: np.ndarray,
    encoder_model: Model,
    decoder_model: Model,
    target_tokenizer,
    max_tgt_len: int,
    beam_width: int = 4,
    length_penalty_alpha: float = 0.7,
) -> str:
    """
    Beam-search decoder for seq2seq inference.
    This reduces greedy-collapse where one generic phrase dominates many inputs.
    """
    start_id, end_id = get_special_ids(target_tokenizer)
    unk_id = target_tokenizer.word_index.get("<unk>")

    state_h, state_c = encoder_model(input_seq, training=False)
    state_h = np.asarray(state_h)
    state_c = np.asarray(state_c)

    beams = [
        {
            "tokens": [start_id],
            "logp": 0.0,
            "h": state_h,
            "c": state_c,
            "ended": False,
        }
    ]

    for _ in range(max_tgt_len):
        candidates = []
        all_ended = True

        for beam in beams:
            if beam["ended"]:
                candidates.append(beam)
                continue

            all_ended = False
            prev_id = int(beam["tokens"][-1])
            target_seq = np.array([[prev_id]], dtype=np.int32)
            dec_probs, next_h, next_c = decoder_model(
                [target_seq, beam["h"], beam["c"]], training=False
            )
            probs = np.asarray(dec_probs)[0, -1, :].astype(np.float64)
            probs[0] = 0.0
            if unk_id is not None and unk_id < probs.size:
                probs[unk_id] = 0.0

            total = probs.sum()
            if total <= 0:
                continue
            probs /= total

            top_ids = np.argsort(probs)[-beam_width:][::-1]
            for tid in top_ids:
                p = float(probs[tid])
                if p <= 0:
                    continue
                token_id = int(tid)
                repeated = token_id in beam["tokens"][-3:]
                # Penalize immediate repetition loops strongly.
                rep_penalty = 2.5 if repeated else 0.0
                candidates.append(
                    {
                        "tokens": beam["tokens"] + [token_id],
                        "logp": beam["logp"] + np.log(p) - rep_penalty,
                        "h": np.asarray(next_h),
                        "c": np.asarray(next_c),
                        "ended": (end_id is not None and token_id == end_id),
                    }
                )

        if not candidates:
            break
        if all_ended:
            break

        # Normalize by length so shorter beams don't always win.
        def norm_score(b):
            out_len = max(len(b["tokens"]) - 1, 1)
            return b["logp"] / (out_len**length_penalty_alpha)

        beams = sorted(candidates, key=norm_score, reverse=True)[:beam_width]

    best = None
    best_score = -np.inf
    for b in beams:
        out_len = max(len(b["tokens"]) - 1, 1)
        score = b["logp"] / (out_len**length_penalty_alpha)
        if score > best_score:
            best_score = score
            best = b

    if best is None:
        return ""

    words = []
    for tid in best["tokens"][1:]:
        if end_id is not None and tid == end_id:
            break
        w = idx_to_word(target_tokenizer, int(tid))
        if not w or w in ("<start>", "<end>", "<unk>"):
            continue
        words.append(w)
    return " ".join(words).strip()


def decode_sequence_translation(input_seq: np.ndarray, assets: dict) -> str:
    # Greedy was generally better for this model; keep it as primary decoding.
    greedy = decode_sequence(
        input_seq,
        assets["encoder_model"],
        assets["decoder_model"],
        assets["hin_tokenizer"],
        assets["max_tgt_len"],
    )
    if not greedy:
        return greedy

    # Only try beam as a rescue path when greedy looks collapsed.
    if is_low_confidence_translation("", greedy):
        beam = decode_sequence_beam(
            input_seq,
            assets["encoder_model"],
            assets["decoder_model"],
            assets["hin_tokenizer"],
            assets["max_tgt_len"],
            beam_width=3,
            length_penalty_alpha=0.6,
        )
        if beam and not is_low_confidence_translation("", beam):
            return beam

    return greedy


def decode_sequence_summarization(input_seq: np.ndarray, assets: dict) -> str:
    return decode_sequence(
        input_seq,
        assets["encoder_model"],
        assets["decoder_model"],
        assets["summary_tokenizer"],
        assets["max_tgt_len"],
    )


def is_low_confidence_translation(source_text: str, translated_text: str) -> bool:
    """
    Heuristic guardrail for degenerate translations.
    Flags very short outputs or heavy token repetition.
    """
    src_words = tokenize_words(source_text) if source_text else []
    out_tokens = translated_text.split()
    if len(src_words) >= 5 and len(out_tokens) < 3:
        return True
    if not out_tokens:
        return True

    unique_ratio = len(set(out_tokens)) / max(len(out_tokens), 1)
    if unique_ratio < 0.45:
        return True

    if len(out_tokens) >= 6:
        bigrams = [
            (out_tokens[i], out_tokens[i + 1]) for i in range(len(out_tokens) - 1)
        ]
        repeat_bigram_count = len(bigrams) - len(set(bigrams))
        if repeat_bigram_count >= 1:
            return True

    return False


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Encoder–Decoder NLP",
        page_icon="🌐",
        layout="centered",
    )

    with st.sidebar:
        st.header("Mode")
        mode = st.selectbox(
            "Choose a task",
            ["Translator Mode", "Summarizer Mode"],
            index=0,
        )

    if mode == "Translator Mode":
        st.title("English → Hindi Translator")
        st.markdown(
            "Enter English text below. The model encodes your sentence and decodes "
            "a Hindi translation using a separate inference decoder (no full-sequence "
            "`predict` on the teacher-forcing graph)."
        )
        st.caption(
            "The English tokenizer keeps the top 10k words; rare words become `<unk>` "
            "in the encoder, which can hurt quality—use common vocabulary when you can."
        )
        user_text = st.text_area("English input", height=160, key="tr_in")
        go = st.button("Generate", key="tr_go")

        if go:
            if not user_text.strip():
                st.warning("Please enter some English text.")
            else:
                with st.spinner("Translating…"):
                    assets = load_translation_assets()
                    seq = encode_and_pad(
                        user_text, assets["eng_tokenizer"], assets["max_src_len"]
                    )
                    out = decode_sequence_translation(seq, assets)
                st.subheader("Hindi output")
                st.write(out if out else "(empty sequence — check tokenization / OOV)")

    else:
        st.title("Text Summarizer")
        st.markdown(
            "Paste a longer piece of text. The summarizer encodes it and generates "
            "a short summary step by step with cached encoder/decoder inference models."
        )
        st.caption(
            "Source text uses the top 8k words; OOV tokens are mapped to `<unk>` before "
            "encoding."
        )
        user_text = st.text_area("Document text", height=220, key="sum_in")
        go = st.button("Generate", key="sum_go")

        if go:
            if not user_text.strip():
                st.warning("Please enter text to summarize.")
            else:
                with st.spinner("Summarizing…"):
                    assets = load_summarizer_assets()
                    seq = encode_and_pad(
                        user_text, assets["text_tokenizer"], assets["max_src_len"]
                    )
                    out = decode_sequence_summarization(seq, assets)
                    used_fallback = False
                    if should_use_fallback_summary(user_text, out):
                        out = extractive_fallback_summary(user_text)
                        used_fallback = True
                st.subheader("Summary")
                st.write(out if out else "(empty sequence — check tokenization / OOV)")


if __name__ == "__main__":
    main()
    