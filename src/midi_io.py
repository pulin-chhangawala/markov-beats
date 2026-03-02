"""
midi_io.py - MIDI file reading and writing

Uses the mido library for MIDI parsing. Handles:
  - Reading MIDI files and extracting note sequences
  - Writing generated sequences back to MIDI
  - Mapping between MIDI note numbers and note names
"""

import mido
import os


# note name mapping (for display)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_name(midi_num):
    """Convert MIDI number to note name (e.g. 60 → C4)."""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


def read_midi(filepath):
    """
    Read a MIDI file and extract note sequences.
    
    Returns a list of (pitch, duration_ticks, velocity) tuples
    for each track that contains note data.
    """
    mid = mido.MidiFile(filepath)
    tracks = []
    
    for i, track in enumerate(mid.tracks):
        notes = []
        active_notes = {}  # pitch -> (start_tick, velocity)
        current_tick = 0
        
        for msg in track:
            current_tick += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start, vel = active_notes.pop(msg.note)
                    duration = current_tick - start
                    if duration > 0:
                        notes.append((msg.note, duration, vel))
        
        if len(notes) > 0:
            tracks.append(notes)
    
    return tracks, mid.ticks_per_beat


def write_midi(notes, filepath, ticks_per_beat=480, tempo_bpm=120, 
               default_velocity=80, default_duration=240):
    """
    Write a sequence of notes to a MIDI file.
    
    Args:
        notes: List of pitch values (ints). Can also be list of
               (pitch, duration, velocity) tuples.
        filepath: Output file path.
        ticks_per_beat: MIDI resolution.
        tempo_bpm: Tempo in BPM.
        default_velocity: Note velocity if not specified.
        default_duration: Note duration in ticks if not specified.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # set tempo
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(mido.MetaMessage('track_name', name='markov-beats', time=0))
    
    for note in notes:
        if isinstance(note, tuple):
            pitch, duration, velocity = note
        else:
            pitch = note
            duration = default_duration
            velocity = default_velocity
        
        # clamp to valid MIDI range
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))
        
        track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))
    
    # end of track
    track.append(mido.MetaMessage('end_of_track', time=0))
    
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    mid.save(filepath)


def midi_info(filepath):
    """Print basic info about a MIDI file."""
    mid = mido.MidiFile(filepath)
    print(f"  File: {filepath}")
    print(f"  Type: {mid.type}")
    print(f"  Ticks/beat: {mid.ticks_per_beat}")
    print(f"  Tracks: {len(mid.tracks)}")
    print(f"  Length: {mid.length:.1f}s")
    
    for i, track in enumerate(mid.tracks):
        notes = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 0)
        print(f"    Track {i}: {track.name or '(unnamed)'} ({notes} notes)")
