import music21
music21.environment.UserSettings()['warnings'] = 0

import copy
def note_octave_mod(note, offset):
    n = copy.deepcopy(note)
    if n.octave is None:
        n.octave = note.pitch.implicitOctave
    n.octave += offset
    return n

def octave_mod(note, offset):
    n = music21.pitch.Pitch(note)
    if n.octave is None:
        n.octave = note.pitch.implicitOctave
    n.octave += offset
    
    return n

def get_all_midi_notes_of_key(keyname,major="major"):
    our_key = music21.key.Key(keyname,major)
    pitches = our_key.pitches
    return [octave_mod(pitch, offset).midi \
            for offset in range(-5,5) \
            for pitch in pitches ]

# all_keys = get_all_keys()
_all_keys = None

def _get_all_keys():
    keys = ["C","F","B-","E-","A-","D-","G-","C-","G","D","A","E","B","F#","C#"]
    majors = ["major","minor"]
    return dict([(key+major,set(get_all_midi_notes_of_key(key,major))) for key in keys for major in majors])

def get_all_keys():
    global _all_keys
    if _all_keys is None:
        _all_keys = _get_all_keys()
    return _all_keys

def in_a_key(notes,all_keys=None):
    snotes = set(notes)
    keys = set()
    if all_keys is None:
        all_keys = get_all_keys()
    for keyname in all_keys:
        keynotes = all_keys[keyname]
        if len(snotes.difference(keynotes)) == 0:
            keys.add(keyname)
    return keys

if __name__ == "__main__":
    print get_all_midi_notes_of_key('C')
    print get_all_keys()
    # not in key
    print(in_a_key([29, 31, 32, 34, 36, 37, 39, 41, 42]))
    # F minor
    print(in_a_key([29, 31, 32, 34, 36, 37, 39, 41]))
    # even more..
    print(in_a_key([29, 31, 32, 34, 36]))
    # even more..
    print(in_a_key([29, 31, 32]))
    # even more..
    print(in_a_key([29, 31]))
