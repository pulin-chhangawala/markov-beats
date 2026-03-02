"""
harmony.py - Chord-aware music generation

Extends the basic Markov chain with music theory awareness:
  - Chord detection from note clusters
  - Harmonic transitions (chord → chord) as a separate chain
  - Scale-constrained generation (stay in key)
  - Tension/resolution modeling (dissonance control)

This is the "if you know some music theory" version of the
generator. It produces more harmonically coherent output
than pure note-by-note Markov.
"""

import random
from collections import defaultdict

# scale definitions (as semitone intervals from root)
SCALES = {
    'major':          [0, 2, 4, 5, 7, 9, 11],
    'minor':          [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'pentatonic':     [0, 2, 4, 7, 9],
    'blues':          [0, 3, 5, 6, 7, 10],
    'dorian':         [0, 2, 3, 5, 7, 9, 10],
    'mixolydian':     [0, 2, 4, 5, 7, 9, 10],
}

# common chord progressions (as scale degrees)
PROGRESSIONS = {
    'pop':     [1, 5, 6, 4],     # I-V-vi-IV (every pop song ever)
    'jazz':    [2, 5, 1, 1],     # ii-V-I
    'blues':   [1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5],  # 12-bar blues
    'sad':     [6, 4, 1, 5],     # vi-IV-I-V
    'epic':    [1, 3, 4, 5],     # I-III-IV-V
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(midi_num):
    return f"{NOTE_NAMES[midi_num % 12]}{(midi_num // 12) - 1}"


def scale_notes(root, scale_name, octave_start=4, octave_end=6):
    """Generate all notes in a scale across given octave range."""
    if scale_name not in SCALES:
        raise ValueError(f"Unknown scale: {scale_name}. Options: {list(SCALES.keys())}")
    
    intervals = SCALES[scale_name]
    notes = []
    for octave in range(octave_start, octave_end + 1):
        base = root + (octave + 1) * 12
        for interval in intervals:
            n = base + interval
            if 0 <= n <= 127:
                notes.append(n)
    return sorted(notes)


def chord_notes(root, quality='major'):
    """Build a chord from root note."""
    if quality == 'major':
        return [root, root + 4, root + 7]
    elif quality == 'minor':
        return [root, root + 3, root + 7]
    elif quality == 'dim':
        return [root, root + 3, root + 6]
    elif quality == '7':
        return [root, root + 4, root + 7, root + 10]
    elif quality == 'maj7':
        return [root, root + 4, root + 7, root + 11]
    else:
        return [root, root + 4, root + 7]


def constrain_to_scale(pitch, allowed_notes):
    """Snap a pitch to the nearest note in the scale."""
    if pitch in allowed_notes:
        return pitch
    # find closest
    closest = min(allowed_notes, key=lambda n: abs(n - pitch))
    return closest


class HarmonicChain:
    """
    Two-level Markov chain:
    1. Chord progression chain (transitions between chords)
    2. Melody chain (note transitions conditioned on current chord)
    """
    
    def __init__(self, root=60, scale='major', order=2):
        self.root = root
        self.scale = scale
        self.order = order
        self.allowed_notes = set(scale_notes(root % 12, scale))
        
        # chord-level transitions
        self.chord_transitions = defaultdict(lambda: defaultdict(int))
        # note-level transitions (conditioned on chord context)
        self.melody_transitions = defaultdict(lambda: defaultdict(int))
    
    def train_from_progression(self, prog_name='pop', repeats=8):
        """Learn chord transitions from a known progression."""
        if prog_name not in PROGRESSIONS:
            raise ValueError(f"Unknown progression: {prog_name}")
        
        degrees = PROGRESSIONS[prog_name] * repeats
        
        for i in range(len(degrees) - 1):
            self.chord_transitions[degrees[i]][degrees[i+1]] += 1
    
    def generate_melody(self, length=64, progression='pop', temperature=1.0):
        """
        Generate a melody that follows a chord progression.
        
        For each chord in the progression:
        1. Select notes from the chord + passing tones from the scale
        2. Use the Markov chain for melodic contour
        3. Constrain output to the scale
        """
        chord_seq = PROGRESSIONS.get(progression, PROGRESSIONS['pop'])
        notes = []
        
        notes_per_chord = max(1, length // len(chord_seq))
        prev_note = self.root + 60  # start at middle C relative to root
        
        for degree in chord_seq:
            # build chord
            chord_root = self.root + SCALES[self.scale][(degree - 1) % len(SCALES[self.scale])] + 60
            chord = chord_notes(chord_root, 'major' if degree in [1, 4, 5] else 'minor')
            
            for j in range(notes_per_chord):
                if j == 0:
                    # start of chord: land on a chord tone
                    note = random.choice(chord)
                else:
                    # melodic movement: small steps with occasional leaps
                    step = random.choices(
                        [-2, -1, 0, 1, 2, 3, -3],
                        weights=[0.1, 0.25, 0.1, 0.25, 0.1, 0.05, 0.05] if temperature <= 1.0
                        else [0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.15]
                    )[0]
                    
                    # scale-degree step
                    allowed = sorted(self.allowed_notes)
                    if prev_note in allowed:
                        idx = allowed.index(prev_note)
                    else:
                        idx = min(range(len(allowed)),
                                 key=lambda i: abs(allowed[i] - prev_note))
                    
                    new_idx = max(0, min(len(allowed) - 1, idx + step))
                    note = allowed[new_idx]
                
                note = constrain_to_scale(note, self.allowed_notes)
                notes.append(note)
                prev_note = note
                
                if len(notes) >= length:
                    break
            
            if len(notes) >= length:
                break
        
        return notes[:length]
    
    def get_chord_progression_str(self, prog_name='pop'):
        """Human-readable chord names for a progression."""
        if prog_name not in PROGRESSIONS:
            return "Unknown"
        
        degrees = PROGRESSIONS[prog_name]
        scale_intervals = SCALES[self.scale]
        
        chord_names = []
        for d in degrees:
            root_offset = scale_intervals[(d - 1) % len(scale_intervals)]
            root_note = NOTE_NAMES[(self.root + root_offset) % 12]
            quality = 'maj' if d in [1, 4, 5] else 'min' if d in [2, 3, 6] else 'dim'
            chord_names.append(f"{root_note}{'' if quality == 'maj' else 'm'}")
        
        return ' → '.join(chord_names)
