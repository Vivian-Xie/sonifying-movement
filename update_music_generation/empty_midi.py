# This file is to empty the origin.mid to prepare for a new round of generation

import mido
def create_empty_midi_file(output_file):
    midi = mido.MidiFile()
    track1 = mido.MidiTrack()
    track2 = mido.MidiTrack()
    track3 = mido.MidiTrack()
    midi.tracks.append(track1)
    midi.tracks.append(track2)
    midi.tracks.append(track3)
    midi.save(output_file)
if __name__ == '__main__':
    output_file = './demo/origin.mid'  
    create_empty_midi_file(output_file)