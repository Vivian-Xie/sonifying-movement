# If you want to modify your own song, upload the song, and turn the melody into notes for the following processes.

import mido

def midi_to_numbers(midi_file):
    midi_data = mido.MidiFile(midi_file)
    note_numbers = []
    for msg in midi_data.tracks[5]:
        if msg.type == 'note_off':  #延长音
            notes_number=128
            note_numbers.append(notes_number)
        if msg.type == 'note_on':  #空音
            if msg.velocity==0:
                notes_number=129
                note_numbers.append(notes_number)
            if msg.velocity!=0:  
                note_number = int(msg.note * 128 / 127)
                note_numbers.append(note_number)

    return note_numbers

# 使用示例
midi_file_path = './demo/Spyro2_countryfarms)_aigei_com.mid'
note_numbers = midi_to_numbers(midi_file_path)
file=open('notes.txt', 'w')
l=len(note_numbers)
print(l)
while l/32-l//32!=0:
    note_numbers.append(129)
    l+=1
for i in note_numbers:
    file.write(str(i) + '\n')
print("Note numbers (0-128):", note_numbers)


