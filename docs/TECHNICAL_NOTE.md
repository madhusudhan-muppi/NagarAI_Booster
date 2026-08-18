# NagarAI — Technical Note

Required one-page disclosure: what we used, what is ours, and what does not work.

## Models and services used

| Component | Model / service | Ours? |
|---|---|---|
| Speech recognition | faster-whisper `small`, int8 CPU | Pre-trained, used as-is |
| Regional speech (Tamil) | AI4Bharat IndicConformer | Pre-trained, used as-is |
| Image classification | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), zero-shot | Pre-trained; **prompt design, ensembling, distractor set and confidence gates are ours** |
| Sentence embeddings | `all-MiniLM-L6-v2` | Pre-trained, used as-is |
| Complaint normalisation | Gemini Flash-Lite via API | Pre-trained; **prompt and precedence logic are ours** |
| Geocoding | OpenStreetMap Nominatim | Public service |
| Map tiles | OpenStreetMap | Public service |

No model was fine-tuned. We have no labelled civic-defect dataset, and we say so
rather than implying training we did not do.

## What is original work

1. **The three-stage deduplication cascade** — category gate, per-category
   geographic radius, then blended text-plus-image similarity, with a
   proximity-dominant shortcut. Union-find merging, incremental by design.
2. **Per-category merge radii.** Waterlogging spreads along a road (200 m); a
   garbage bin is a point (60 m). One global radius is wrong for both.
3. **The priority model** `S² × (1 + ln N) × (1 + D/7) × P`, the two-lane
   hazard/routine split, and the gaming-resistance argument behind each term.
4. **Hybrid classification with an explicit precedence policy** — vision, then
   LLM, then a domain keyword table covering Tamil and Hindi transliteration —
   plus a hazard safety override.
5. **Graceful degradation throughout.** Every model is optional; the system runs
   with zero external dependencies and no network.

## Measured results

15-complaint judging set, same thresholds, only the embedder changed:

| Embedder | Distinct issues | Merges |
|---|---|---|
| TF-IDF (dependency-free fallback) | 9 | 6 |
| all-MiniLM-L6-v2 | 8 | 7 |

The deciding pair is C011/C012 — two waterlogging reports 173 m apart sharing
almost no vocabulary. Sentence embeddings score them 0.45 against a 0.42
threshold.

## Known failure modes

1. **The 0.42 threshold has a 0.03 margin.** C011/C012 is the only merge in the
   set decided by the semantic stage rather than by proximity or location-text
   overlap. Fifteen samples cannot validate a threshold. We report this rather
   than lowering the threshold to look safer.

2. **A frontier LLM lost to a keyword table on a safety-critical case.**
   Gemini translated the Tamil `kambi kashtama irukku` as "difficulties with
   metal wire or mesh" and classified it `other` — literally correct, civically
   useless. Our keyword table matched `kambi` and returned `live_wire`. We now
   treat an LLM `other` as an abstention rather than a claim, and let a hazard
   keyword hit override a non-hazard classification. Found by testing our own
   language against our own system.

3. **CLIP zero-shot classification is confidence-gated because it was wrong.**
   Our first prompt set sent a pothole photo to `other`. We added prompt
   ensembling, explicit non-civic distractors, and minimum probability and
   margin gates. Below those gates vision abstains rather than overriding text.

4. **Severity estimation is coarse.** It comes from a per-category table, not
   from measurement. Estimating pothole extent from one photo with no reference
   object is not a solved problem and we do not claim to have solved it.

5. **Conflicting modalities are detected but not resolved.** A pothole photo
   with a garbage voice note files as one complaint under the vision category,
   with the losing signal recorded as `secondary_category` and flagged on the
   dashboard. A ward officer decides; the system does not.

6. **Text geocoding needs the network.** GPS share and photo EXIF work offline;
   landmark lookup via Nominatim does not, and degrades to no map position.

7. **Image similarity only helps when both complaints have photos.** Roughly
   half of real complaints are text-only, so the blended term is inactive for
   them and the text signal carries the merge alone.

8. **No adversarial testing at scale.** Our gaming resistance is structural —
   unique reporters, logarithmic damping, severity from the image, a capped
   ageing term — but untested against a real coordinated campaign.

## Privacy

Telegram user IDs are hashed before storage; we count unique reporters without
holding identities. Complaint text is sent to a third-party LLM API for
normalisation, which is a real disclosure concern for citizen data — a
production deployment would use a paid tier with no training rights, or a local
model. The demo uses synthetic complaints only.