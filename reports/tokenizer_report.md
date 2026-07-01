# Tokenizer Report
## Corpus
- File: `data/model/train.txt`
- Characters: 389,157
- Words: 63,682

## Statistics
**Character tokenizer**
- Vocabulary size: 115
- Tokens for train.txt: 389,157
- Characters in train.txt: 389,157
- Compression ratio (tokens/char): 1.0000
- Average tokens per word: 5.09
- Encoding time: 10.1 ms

**BPE tokenizer (num_merges=1000)**
- Vocabulary size: 1,065
- Tokens for train.txt: 137,535
- Characters in train.txt: 389,157
- Compression ratio (tokens/char): 0.3534
- Average tokens per word: 2.22
- Encoding time: 17549.1 ms

## Round-trip examples
### Character tokenizer
Input:   `Flinders Street Station was built in 1905.`
IDs:     [41, 75, 72, 77, 67, 68, 81, 82, 5, 54, 83, 81, 68, 68, 83, 5, 54, 83, 64, 83]...
Decoded: `Flinders Street Station was built in 1905.`
Round-trip: ✓

Input:   `The Eastern Market in Melbourne was demolished in 1960.`
IDs:     [55, 71, 68, 5, 40, 64, 82, 83, 68, 81, 77, 5, 48, 64, 81, 74, 68, 83, 5, 72]...
Decoded: `The Eastern Market in Melbourne was demolished in 1960.`
Round-trip: ✓

Input:   `Located at 309 Bourke Street, the hotel opened in 1883.`
IDs:     [47, 78, 66, 64, 83, 68, 67, 5, 64, 83, 5, 24, 21, 30, 5, 37, 78, 84, 81, 74]...
Decoded: `Located at 309 Bourke Street, the hotel opened in 1883.`
Round-trip: ✓

Input:   `Carlton, Fitzroy and Collingwood are inner Melbourne suburbs.`
IDs:     [38, 64, 81, 75, 83, 78, 77, 17, 5, 41, 72, 83, 89, 81, 78, 88, 5, 64, 77, 67]...
Decoded: `Carlton, Fitzroy and Collingwood are inner Melbourne suburbs.`
Round-trip: ✓

Input:   `Heritage Victoria protects significant buildings across the state.`
IDs:     [43, 68, 81, 72, 83, 64, 70, 68, 5, 57, 72, 66, 83, 78, 81, 72, 64, 5, 79, 81]...
Decoded: `Heritage Victoria protects significant buildings across the state.`
Round-trip: ✓

### BPE tokenizer
Input:   `Flinders Street Station was built in 1905.`
IDs:     [92, 171, 168, 295, 1013, 333, 575, 22, 30, 21, 26, 19]
Decoded: `Flinders Street Station was built in 1 9 0 5 .`
Round-trip: ✓

Input:   `The Eastern Market in Melbourne was demolished in 1960.`
IDs:     [180, 83, 123, 575, 128, 1013, 385, 575, 22, 30, 27, 21, 19]
Decoded: `The Eastern Market in Melbourne was demolished in 1 9 6 0 .`
Round-trip: ✓

Input:   `Located at 309 Bourke Street, the hotel opened in 1883.`
IDs:     [118, 724, 291, 289, 24, 21, 30, 56, 171, 17, 942, 537, 755, 575, 22, 29, 29, 24, 19]
Decoded: `Located at 3 0 9 Bourke Street , the hotel opened in 1 8 8 3 .`
Round-trip: ✓

Input:   `Carlton, Fitzroy and Collingwood are inner Melbourne suburbs.`
IDs:     [62, 262, 639, 958, 17, 91, 253, 72, 639, 588, 1011, 747, 272, 574, 707, 128, 926, 19]
Decoded: `Carlton , Fitzroy and Collingwood are inner Melbourne suburbs .`
Round-trip: ✓

Input:   `Heritage Victoria protects significant buildings across the state.`
IDs:     [100, 191, 833, 930, 417, 879, 895, 258, 332, 215, 872, 942, 906, 290, 19]
Decoded: `Heritage Victoria protects significant buildings across the state .`
Round-trip: ✓

## Unknown-token behaviour
Input: `τελεία 日本語 émoji 🏛️`

| Tokenizer | UNK ids | UNK rate |
| --- | --- | --- |
| Char | 11 | 58% |
| BPE | 11 | 41% |

## Saved artifacts
- `data/model/tokenizers/char_tokenizer.json` (1,592 bytes)
- `data/model/tokenizers/bpe_tokenizer.json` (57,540 bytes)
