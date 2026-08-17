# Literature Survey and Research Gap

> Verify every URL and add an access date before submission. Do not submit a
> reference you have not personally opened.

## 1. Existing civic grievance systems

### 1.1 India

**CPGRAMS** (Centralised Public Grievance Redress and Monitoring System, DARPG).
The national grievance portal. Citizens file free text, which is manually routed
to a ministry or department. Text-only, English and Hindi, no deduplication, no
automated prioritisation. Ranking is by escalation and ageing, not severity.

**Swachhata App** (Ministry of Housing and Urban Affairs).
Photo-based sanitation complaints with GPS tagging and a resolution photo at
closure. Establishes photo-plus-location intake as workable at national scale.
Categories are chosen manually by the citizen from a dropdown; identical
complaints from the same street are filed and worked as separate tickets.

**Namma Chennai / Chennai Corporation grievance redressal** (GCC).
Ward-level complaint filing with department routing and SLA timers. Regional
relevance for this project. Intake is form-based; no voice, no clustering.

**BBMP Sahaaya** (Bruhat Bengaluru Mahanagara Palike).
Ward-wise complaint tracking with public dashboards. Notable for transparency of
pending counts, which is a useful precedent for our official dashboard.

### 1.2 International

**FixMyStreet** (mySociety, UK). Open source. Map-first reporting with pin
placement. Detects nearby existing reports and invites the citizen to confirm
rather than re-file — the closest existing system to our deduplication goal, but
the match is by map proximity alone with a human making the final call, not by
semantic similarity.

**SeeClickFix** (USA). Commercial 311 platform with duplicate suggestion and
department routing. Proximity and keyword based.

**NYC 311 / BOS:311**. Large-scale municipal request systems. Well-studied for
reporting bias: complaint volume tracks civic engagement and demographics, not
actual problem severity. This is the strongest published justification for our
decision to damp headcount logarithmically and take severity from the image
rather than from complaint counts.

## 2. Enabling techniques

| Area | Method | Reference |
|---|---|---|
| Speech recognition | Whisper, weakly supervised multilingual ASR | Radford et al., 2022 |
| Indic speech | AI4Bharat IndicWav2Vec / IndicConformer | AI4Bharat, IIT Madras |
| Sentence embeddings | Sentence-BERT siamese architecture | Reimers & Gurevych, EMNLP 2019 |
| Multilingual embeddings | LaBSE, language-agnostic sentence embeddings | Feng et al., 2020 |
| Image classification | CLIP, zero-shot transfer from natural language supervision | Radford et al., ICML 2021 |
| Object detection | YOLO family, single-stage real-time detection | Redmon et al., 2016 onward |
| Density clustering | DBSCAN, density-based spatial clustering | Ester et al., KDD 1996 |
| Geo-semantic clustering | ST-DBSCAN, spatial-temporal extension | Birant & Kut, 2007 |

## 3. Research gap

Each capability exists in isolation. No deployed Indian civic system combines them.

**Gap 1 — Intake is unimodal and monolingual.**
Existing systems accept text (CPGRAMS) or photo (Swachhata), never a voice note
in Tamil accompanied by a sideways photo and a location share. A citizen who is
not comfortable typing English is effectively excluded from the grievance system.

**Gap 2 — Deduplication is proximity-only or absent.**
FixMyStreet matches on map distance; CPGRAMS does not match at all. Nobody
combines semantic similarity of the complaint text with geographic distance and
image similarity. Proximity alone cannot tell a pothole from a garbage pile two
metres away; semantics alone cannot tell one pothole from an identically worded
pothole in another ward. Both signals are necessary and neither is sufficient.

**Gap 3 — Prioritisation is opaque or volume-driven.**
Ranking follows escalation, ageing, or raw complaint count. A count-driven
ranking rewards the loudest and best-connected neighbourhoods, which the 311
literature documents directly. No system publishes its ranking formula to the
officials using it.

**Gap 4 — Nothing is gaming-resistant.**
Where complaint count drives priority, an organised group can promote a minor
issue. No published municipal system addresses this.

## 4. What this project contributes

1. A single intake path accepting Tamil/Hindi/English voice, photo, and text in
   any combination, normalised to one structured complaint record.
2. A three-stage deduplication cascade — category gate, per-category geographic
   radius, then semantic similarity — with a proximity-dominant shortcut, and a
   recorded human-readable reason for every merge.
3. A published priority formula, `S² × (1 + ln N) × (1 + D/7) × P`, with severity
   superlinear and headcount logarithmic, plus a separate hazard lane, designed
   so that a 40-report pothole cannot outrank a 2-report live wire.
4. An explicit gaming-resistance argument: unique reporters rather than complaint
   count, severity from the vision model rather than citizen adjectives, and a
   capped ageing term.

## 5. Comparison table

| System | Voice | Photo | Regional lang | Semantic dedup | Explainable priority | Gaming-resistant |
|---|---|---|---|---|---|---|
| CPGRAMS | No | No | Partial | No | No | No |
| Swachhata | No | Yes | Partial | No | No | No |
| Namma Chennai | No | Yes | Partial | No | No | No |
| FixMyStreet | No | Yes | n/a | Proximity only | No | No |
| SeeClickFix | No | Yes | n/a | Proximity + keyword | No | No |
| **NagarAI** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
