# Corpora

Both corpora are public. The scripts do not download anything; they read a local parquet file and
write a character-id archive.

## 1. United Nations Parallel Corpus (en-zh), the main corpus

Source: United Nations Parallel Corpus v1.0, English-Chinese portion, published by the United
Nations. Refer to the corpus page for the current download location and for the terms of use.

Conversion into the layout the scripts read:

```
column  "translation" : struct<en: string, zh: string>
```

One row per sentence pair; no other column is used. The archive we build holds 608k pairs after the
length filter below, split 596k train / 12,060 validation (paper §4).

Filtering rule, applied exactly as in `src/data.py`:

```
2 <= len(en) <= 100 characters      # the English window is 100 positions
2 <= len(zh) <= 32  characters      # the Chinese window is 32 positions
```

## 2. OPUS-100 (en-zh), the second corpus

Source: OPUS-100, `en-zh` configuration, training split, available from the Hugging Face Hub
(`Helsinki-NLP/opus-100`). The script expects the parquet shard, for example `train-00000-of-00001.parquet`:

```
huggingface-cli download Helsinki-NLP/opus-100 --repo-type dataset --include "*en-zh*" --local-dir opus100
```

After the same filtering the archive holds 603,836 pairs, split 591,145 train / 12,691 validation.

The two corpora differ in Chinese-side conditional entropy by 0.37 to 1.15 nat depending on the
length bucket, which is why the paper never compares absolute metrics across them (paper §4, §5.7).

## 3. The character-id archive

Both converters write the same keys:

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `en` | (N, 100) | int16 | English character ids, right-aligned in a fixed 100-position window |
| `zh` | (N, 32) | int16 | Chinese character ids, left-aligned |
| `en_len` | (N,) | int8 | number of valid English positions |
| `zh_len` | (N,) | int8 | number of valid Chinese positions |
| `split` | (N,) | int8 | 0 train, 1 validation |
| `es` | (V,) | <U1 | English vocabulary, most frequent first |
| `zs` | (V,) | <U1 | Chinese vocabulary, most frequent first |

Ids: 0 is padding, 1 is unknown, symbol `i` of a vocabulary is `i + 2`.

The archives used in the paper end up with 2048 Chinese and 153 English symbols. The English table is
requested at 200 entries but the filtered corpus contains fewer distinct symbols; the training and
evaluation scripts read both vocabulary sizes from the archive, so a rebuild on a different corpus
changes the numbers without breaking the shapes.

## 4. The reduced archive for the single-board benchmark

2 GB of RAM cannot hold the full archive (about 650 MB once the int64 arrays are materialised), and
the board reads from an SD card. `src/make_pi_data.py` writes a 600-row test-only archive with the
same vocabularies and the same shapes, 0.04 MB in total. It is what `src/pi_bench.py` reads, and it
does not change any result: the benchmark times decoding, not accuracy.
