# T1 validation gap preferences across decoder architectures

Compare ridge, CNN + ELU, CNN–BiLSTM, and EEGNet regression across the
same 12 within-participant train and validation splits used in T_8.

- Use series 1–6 to train and series 7 to select end gaps. Use the
  original T1 common target indices, 128 × 32 windows, training-only
  scaling, and six gaps of 2, 50, 100, 200, 500, and 1,000 ms.
- Reuse the verified T1 ridge scores with validation-selected penalties
  and the seed-42 CNN + ELU validation scores. The earlier three-seed
  CNN + ELU counts are a separate sensitivity comparison.
- Train CNN–BiLSTM and EEGNet regression using their existing project
  architectures, with seed 42, Adam/MSE, batch size 64, and the final
  15-epoch checkpoint for each fit. Match initial weights across gaps
  within each architecture and participant. Use one model per
  participant, gap and architecture, for 144 new fits on GPU.
- For each participant and decoder, select the gap with the lowest
  full-precision validation XYZ MSE, breaking exact ties toward the
  shorter gap. Record the second-best score and the margin. Each
  participant contributes one count to each architecture histogram.
- Compare the most frequent gap or tied gaps per decoder. Summed counts
  across decoders are descriptive because each participant contributes
  to four bars in the combined distribution. Retain per-participant
  validation scores and verify new saved predictions against the T1
  validation target indices and the recorded scores before reporting.

The outcome describes validation preferences for these models and this
split. New decoders are evaluated on validation series 7 only. The
single neural seed and single validation series limit how precisely the
preferred gap can be inferred. A selected gap is a property of the
predictive comparison.

