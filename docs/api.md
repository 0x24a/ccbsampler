# API reference

The Rust client ccbsampler-client is a thin CLI wrapper. It parses the classic UTAU resampler arguments, converts the pitch string, and POSTs everything as JSON to the server. Editors that talk to the server directly can use the same HTTP API.

## `POST /resample`

Request body (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `in_file` | string | - | Path to the source sample. |
| `out_file` | string | - | Path to the target to write. The special value `"nul"` skips writing. |
| `pitch` | string | - | Target pitch, e.g. `"C4"`, `"A#3"`. |
| `velocity` | float | - | Consonant speed. |
| `flags` | object | `{}` | UTAU flags, see the table below. |
| `offset` | float | `0.0` | Start of the usable region, in ms. |
| `length` | int | `1000` | Required note length, in ms. |
| `consonant` | float | `0.0` | Fixed (consonant) region length, in ms. |
| `cutoff` | float | `0.0` | End of the usable region, in ms from the end. |
| `volume` | float | `100.0` | Accepted but not used in rendering. (TODO) |
| `modulation` | float | `0.0` | Accepted but not used in rendering. (TODO) |
| `tempo` | float | `100.0` | Tempo used for pitchbend timing. |
| `pitchbend` | float[] | `[]` | Pitch curve in cents relative to `pitch`. Same format as the original UTAU. |

Response body (JSON):

| Field | Type | Description |
|---|---|---|
| `status` | `"ok"` \| `"error"` | Render outcome. |
| `out_file` | string \| null | Echoes the requested output path on success. |
| `error` | string \| null | Error message on failure. |
| `metrics` | object \| null | Per-render timing breakdown, see below. |

Render metrics:

| Field | Meaning |
|---|---|
| `feature_ms` | Mel extraction (cache miss) or cache load time. Near zero on cache hits. |
| `queue_ms` | Time spent waiting for a free slot in the render queue. |
| `infer_ms` | Vocoder inference + post-processing. |
| `total_ms` | Wall time of the whole render. |

A typical server log line summarizes the same numbers:

```
do.wav → do.wav.out   feature=  12ms  queue=   0ms  infer= 210ms  total= 240ms
```

## UTAU flags

The server accepts these flags via the `flags` object (`{"g": 10}` or `{"He": true}` for switch-type flags). The schema (`schemas.py`) accepts all of the keys below; the ones marked "not implemented" are validated but currently ignored by the renderer.

| Flag | Effect |
|---|---|
| `g` | Gender factor. −600 to 600. |
| `t` | Pitch offset in cents. |
| `A` | Pitch-bend amplitude envelope. −100 to 100. |
| `P` | Loudness normalization strength (0–100, default 100). Requires `wave_norm` to be enabled in `config.yaml`. |
| `Hb` | hnsep: breathiness mixing. Requires hnsep to be enabled in `config.yaml`. |
| `Hv` | hnsep: harmonic mixing. Requires hnsep to be enabled in `config.yaml`. |
| `Ht` | hnsep: harmonic tension. Requires hnsep to be enabled in `config.yaml`. |
| `He` | Enable loop mode for this note. |
| `G` | Regenerate features, bypassing the cache. |
| `fe`, `fl`, `fo`, `fv`, `fp`, `ve`, `vo`, `B`, `S`, `p`, `R`, `D`, `C`, `Z` | Accepted but not implemented. (TODO) |

## Caching

Feature extraction results are cached next to the source wav as `{stem}_Hb{n}_Hv{n}_Ht{n}_g{n}.hifi.npz`. The cache key includes the hnsep flags and gender value, so different flag combinations each get their own file. A cache hit makes a extremely low `feature_ms`, and the `G` flag forces a regeneration. Rendering the same note twice only redoes the vocoder work.