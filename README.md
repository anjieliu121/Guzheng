# Guzheng
# MIDI
* Human-reviewed
* No duplicated notes with the same pitch, start tick, and similar duration
* No non-pentatonic notes played in glissando
* No incorrect notes based on the sheets
* Each music piece is transposed into at most 5 pentatonic scales (D, G, F, C, and A).
  * If the transposed file has a range outside the guzheng’s compass for that scale, then the file is disregarded.
  * We acknowledge that transposition is an approximation, not a substitute for real musical diversity. A real G pentatonic guzheng piece sounds different from a D pentatonic piece transposed up a fifth — the phrasing tendencies, the characteristic ornaments, and the idiomatic gestures differ by key because the physical layout of the guzheng strings changes with tuning. Augmentation via transposition increases pitch diversity in the training data but does not capture key-specific performance characteristics.

# Resources
* [Wikepedia: Numbered Musical Notation](https://en.wikipedia.org/wiki/Numbered_musical_notation)
* [Inspiration of how to note jianpu in plain text](https://github.com/RobertWinslow/jianpu-ascii-font)
* [English names of some guzheng pieces](https://chinesemusics.com/en_us/info-db/p-notes/guzheng-pn/)
* [English names and origins of some guzheng pieces](https://www.lcsd.gov.hk/CE/CulturalService/Programme/pdf/program_note1_433_2.pdf)

