"""
markov.py - Markov chain for music generation

Builds a state transition matrix from MIDI note sequences. Each state
is a tuple of N consecutive notes (N-gram), and transitions represent
the probability of the next note given the current context.

Higher-order chains (bigger N) sound more like the training data but
are less creative. N=2 or N=3 is usually the sweet spot.
"""

import random
from collections import defaultdict


class MarkovChain:
    """N-gram Markov chain for sequential data."""
    
    def __init__(self, order=2):
        """
        Args:
            order: Number of previous notes to condition on.
                   Higher = more similar to training data.
        """
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(int))
        self._trained = False
    
    def train(self, sequences):
        """
        Train on a list of note sequences.
        
        Each sequence is a list of (pitch, duration, velocity) tuples.
        We build the chain on pitch values and carry duration/velocity
        as metadata.
        """
        for seq in sequences:
            pitches = [note[0] for note in seq]
            
            for i in range(len(pitches) - self.order):
                state = tuple(pitches[i:i + self.order])
                next_note = pitches[i + self.order]
                self.transitions[state][next_note] += 1
        
        self._trained = True
        total_states = len(self.transitions)
        total_transitions = sum(
            sum(v.values()) for v in self.transitions.values())
        print(f"  Trained: {total_states} states, {total_transitions} transitions")
    
    def _weighted_choice(self, state):
        """Pick next note probabilistically from transition counts."""
        if state not in self.transitions:
            # unknown state: pick a random known state
            state = random.choice(list(self.transitions.keys()))
        
        options = self.transitions[state]
        total = sum(options.values())
        r = random.uniform(0, total)
        
        cumulative = 0
        for note, count in options.items():
            cumulative += count
            if r <= cumulative:
                return note
        
        return list(options.keys())[-1]
    
    def generate(self, length=64, seed_state=None, temperature=1.0):
        """
        Generate a sequence of notes.
        
        Args:
            length: Number of notes to generate.
            seed_state: Starting state (tuple). Random if None.
            temperature: Controls randomness. <1 = more predictable,
                        >1 = more random. 1.0 = unmodified probabilities.
        
        Returns:
            List of pitch values.
        """
        if not self._trained:
            raise RuntimeError("Train the chain first!")
        
        if seed_state is None:
            seed_state = random.choice(list(self.transitions.keys()))
        
        result = list(seed_state)
        current_state = seed_state
        
        for _ in range(length):
            if current_state not in self.transitions:
                current_state = random.choice(list(self.transitions.keys()))
            
            if temperature != 1.0:
                next_note = self._temperature_choice(current_state, temperature)
            else:
                next_note = self._weighted_choice(current_state)
            
            result.append(next_note)
            current_state = tuple(result[-self.order:])
        
        return result
    
    def _temperature_choice(self, state, temperature):
        """
        Temperature-scaled sampling. Lower temperature makes the chain
        more deterministic (always picks the most likely next note).
        Higher temperature makes it more random.
        """
        import math
        
        options = self.transitions[state]
        total = sum(options.values())
        
        # apply temperature scaling to probabilities
        scaled = {}
        for note, count in options.items():
            prob = count / total
            scaled[note] = math.exp(math.log(prob + 1e-10) / temperature)
        
        scaled_total = sum(scaled.values())
        r = random.uniform(0, scaled_total)
        
        cumulative = 0
        for note, weight in scaled.items():
            cumulative += weight
            if r <= cumulative:
                return note
        
        return list(options.keys())[-1]
    
    def get_stats(self):
        """Return chain statistics."""
        n_states = len(self.transitions)
        avg_options = sum(len(v) for v in self.transitions.values()) / max(n_states, 1)
        total_trans = sum(sum(v.values()) for v in self.transitions.values())
        
        return {
            'order': self.order,
            'num_states': n_states,
            'total_transitions': total_trans,
            'avg_options_per_state': round(avg_options, 1),
        }
