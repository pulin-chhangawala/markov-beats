# markov-beats

Music generation using Markov chains. Learns note transition patterns from MIDI files, then generates new melodies that sound like the training data but are entirely new compositions.

Started this while procrastinating on a probability problem set and realized Markov chains are way more fun when they make sound instead of converging to stationary distributions.

## Quick Start

```bash
pip install -r requirements.txt

# generate from built-in demo patterns
python src/generate.py --demo -o output/demo.mid

# train on your own MIDI files
python src/generate.py -i songs/chopin.mid -o output/chopin_style.mid

# control creativity
python src/generate.py --demo --temperature 0.5   # conservative
python src/generate.py --demo --temperature 2.0   # experimental

# change chain order (context window)
python src/generate.py --demo --order 3 --length 256
```

## How It Works

1. **Parse** MIDI files into note sequences (pitch, duration, velocity)
2. **Build** an N-gram transition table: for every N consecutive notes, record what comes next
3. **Generate** new notes by random walking through the chain, where each next note is sampled from the learned probability distribution
4. **Write** the result back to MIDI

### Temperature Control

The `temperature` parameter scales the probability distribution before sampling:
- **Low temperature (0.1-0.5)**: Always picks the most likely next note. Sounds "safe" and predictable
- **Temperature = 1.0**: Unmodified probabilities. Balanced
- **High temperature (1.5-3.0)**: Flattens probabilities, making unlikely transitions more common. Sounds more "creative" (or chaotic)

### Chain Order

The `order` parameter controls how many previous notes the chain considers:
- **Order 1**: Only looks at the current note. Very random
- **Order 2**: Looks at the last 2 notes. Sweet spot for melody
- **Order 3+**: More context, sounds closer to the original. Risk of just replaying training data verbatim

## Project Structure

```
src/
├── markov.py       # Core Markov chain with temperature sampling
├── midi_io.py      # MIDI file reading/writing (via mido)
└── generate.py     # CLI: training, generation, demo patterns
```

## Requirements

- Python 3.8+
- mido (MIDI library)

## File Format

Input: Standard MIDI files (.mid, .midi). Multi-track supported (each track is a separate training sequence).

Output: Type 0 MIDI file with a single track.

## Limitations

- Monophonic only (no chords, as that would need a different state representation)
- No rhythm modeling (durations are borrowed from training data or fixed)
- Small corpus = repetitive output. Works best with 3+ MIDI files of similar style
