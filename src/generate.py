"""
generate.py - Generate music using a Markov chain trained on MIDI files

Usage:
    # generate from a MIDI file:
    python generate.py --input songs/bach.mid --output output/generated.mid

    # generate from the built-in demo patterns:
    python generate.py --demo --output output/demo.mid

    # control creativity:
    python generate.py --demo --temperature 0.5    # more predictable
    python generate.py --demo --temperature 2.0    # more chaotic
"""

import argparse
import os
import sys

from markov import MarkovChain
from midi_io import read_midi, write_midi, midi_info, note_name


def create_demo_sequences():
    """
    Generate some simple musical sequences for testing without
    needing external MIDI files.

    These are basic chord progressions and melodies in C major.
    """
    # C major scale patterns
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5

    sequences = []

    # ascending/descending scale
    scale_up_down = [(n, 240, 80) for n in c_major + list(reversed(c_major[:-1]))]
    sequences.append(scale_up_down * 4)

    # I-IV-V-I chord arpeggios
    I  = [60, 64, 67, 72]   # C major
    IV = [65, 69, 72, 77]   # F major
    V  = [67, 71, 74, 79]   # G major

    arpeggio = []
    for chord in [I, IV, V, I] * 4:
        for note in chord:
            arpeggio.append((note, 240, 80))
        for note in reversed(chord):
            arpeggio.append((note, 120, 60))
    sequences.append(arpeggio)

    # simple melody (twinkle twinkle style)
    melody_notes = [
        60, 60, 67, 67, 69, 69, 67,
        65, 65, 64, 64, 62, 62, 60,
        67, 67, 65, 65, 64, 64, 62,
        67, 67, 65, 65, 64, 64, 62,
        60, 60, 67, 67, 69, 69, 67,
        65, 65, 64, 64, 62, 62, 60,
    ]
    sequences.append([(n, 480, 90) for n in melody_notes])

    # bluesy pattern (C minor pentatonic)
    blues = [60, 63, 65, 66, 67, 70, 72]
    blues_lick = []
    for _ in range(8):
        import random
        random.seed(42)  # deterministic for reproducibility
        for _ in range(8):
            note = random.choice(blues)
            dur = random.choice([120, 240, 480])
            vel = random.randint(60, 100)
            blues_lick.append((note, dur, vel))
    sequences.append(blues_lick)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate music with Markov chains')
    parser.add_argument('--input', '-i', nargs='+', help='Input MIDI file(s)')
    parser.add_argument('--output', '-o', default='output/generated.mid',
                        help='Output MIDI file')
    parser.add_argument('--demo', action='store_true',
                        help='Use built-in demo patterns instead of input files')
    parser.add_argument('--order', type=int, default=2,
                        help='Markov chain order (default: 2)')
    parser.add_argument('--length', type=int, default=128,
                        help='Number of notes to generate (default: 128)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Creativity control: <1 predictable, >1 chaotic')
    parser.add_argument('--tempo', type=int, default=120,
                        help='Output tempo in BPM')
    parser.add_argument('--info', action='store_true',
                        help='Print info about input MIDI file and exit')
    args = parser.parse_args()

    if args.info and args.input:
        for f in args.input:
            midi_info(f)
        return

    # get training sequences
    if args.demo:
        print("Using demo patterns...")
        sequences = create_demo_sequences()
    elif args.input:
        sequences = []
        for filepath in args.input:
            print(f"Reading {filepath}...")
            tracks, tpb = read_midi(filepath)
            sequences.extend(tracks)
            print(f"  Found {len(tracks)} track(s)")
    else:
        print("Error: provide --input MIDI files or use --demo")
        sys.exit(1)

    if not sequences:
        print("Error: no note data found")
        sys.exit(1)

    total_notes = sum(len(s) for s in sequences)
    print(f"  Total training notes: {total_notes}")

    # train Markov chain
    print(f"\nTraining (order={args.order})...")
    chain = MarkovChain(order=args.order)
    chain.train(sequences)

    stats = chain.get_stats()
    print(f"  States: {stats['num_states']}")
    print(f"  Avg options/state: {stats['avg_options_per_state']}")

    # generate
    print(f"\nGenerating {args.length} notes (temperature={args.temperature})...")
    pitches = chain.generate(length=args.length, temperature=args.temperature)

    # show first few notes
    note_str = ' '.join(note_name(p) for p in pitches[:16])
    print(f"  First 16 notes: {note_str} ...")

    # write output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    write_midi(pitches, args.output, tempo_bpm=args.tempo)
    print(f"\n  Output: {args.output}")
    print(f"  Open with any MIDI player (e.g., VLC, GarageBand, MuseScore)")


if __name__ == '__main__':
    main()
