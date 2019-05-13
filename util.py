from nltk.util import ngrams
from nltk.corpus import gutenberg
import nltk
import glob
import argparse
import random
from random import shuffle

from mido import Message, MidiFile, MidiTrack

def load_split_file(filename):
    tokens = open(filename).read().strip().replace("\n"," ").split(" ")
    return [token for token in tokens if token != ""]

def get_corpus():
    filenames = glob.glob('./corpus.txt/*.txt')
    songs = [load_split_file(filename) for filename in filenames]
    return songs

n_for_ngrams = 3
right_pad_symbol = 'EOS'
left_pad_symbol  = 'BOS'

def our_ngrams(sentence, our_n=None):
    if our_n is None:
       our_n = n_for_ngrams
    return ngrams(sentence, our_n, pad_left = True, pad_right = True, right_pad_symbol=right_pad_symbol, left_pad_symbol=left_pad_symbol)

# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2014 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Daniel Blanchard <dblanchard@ets.org>
#          Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# http://www.nltk.org/_modules/nltk/model/ngram.html
# Copyright (C) 2001-2018 NLTK Project
# 
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
def cross_entropy(estimator,text,n=n_for_ngrams,lpad=left_pad_symbol,rpad=right_pad_symbol):
    """
    Calculate the approximate cross-entropy of the n-gram model for a
    given evaluation text.
    This is the average log probability of each word in the text.

    :param text: words to use for evaluation
    :type text: list(str)
    """

    e = 0.0
    text = [lpad] + text + [rpad]
    for i in range(n - 1, len(text)):
        #context = tuple(text[i - n + 1:i])
        context = tuple(text[i - n + 1:i+1])
        #token = text[i]
        estimate = estimator.logprob(context)
        #print((context,estimate))
        e += estimate
    return -1 * e / float(len(text) - (n - 1))


def probability_sum(our_grams, estimator):
    for gram in our_grams:
        prob_sum += estimator.prob(gram)
    return prob_sum


def apply_vocab(from_vocab=None,to_vocab=None,text=None):
    assert(to_vocab is not None)
    assert(from_vocab is not None)
    assert(text is not None)
    assert(len(to_vocab) == len(from_vocab))
    vmap = dict(zip(from_vocab,to_vocab))
    #print(vmap)
    return ([vmap[x] for x in text], vmap)

def render_midifile( tokens, midi_name="midi",note_length=256 ):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=12, time=0))
    for token in tokens:
        note = int(token[1:])
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=127, time=note_length))
    mid.save('outputs/' + midi_name + '.mid')

def render_textfile(notes, notes_name):
    fd = file('outputs/' + notes_name + '.txt','w')
    fd.write(" ".join(notes))
    fd.close()

def render_notesfile(notes, notes_name):
    fd = file('outputs/' + notes_name + '.notes','w')
    fd.write(tokens_to_notes_full(notes))
    fd.close()

midi_to_note_dict = {
    "127":"G9",
    "126":"Gb9",
    "125":"F9",
    "124":"E9",
    "123":"Eb9",
    "122":"D9",
    "121":"Db9",
    "120":"C9",
    "119":"B8",
    "118":"Bb8",
    "117":"A8",
    "116":"Ab8",
    "115":"G8",
    "114":"Gb8",
    "113":"F8",
    "112":"E8",
    "111":"Eb8",
    "110":"D8",
    "109":"Db8",
    "108":"C8",
    "107":"B7",
    "106":"Bb7",
    "105":"A7",
    "104":"Ab7",
    "103":"G7",
    "102":"Gb7",
    "101":"F7",
    "100":"E7",
    "99":"Eb7",
    "98":"D7",
    "97":"Db7",
    "96":"C7",
    "95":"B6",
    "94":"Bb6",
    "93":"A6",
    "92":"Ab6",
    "91":"G6",
    "90":"Gb6",
    "89":"F6",
    "88":"E6",
    "87":"Eb6",
    "86":"D6",
    "85":"Db6",
    "84":"C6",
    "83":"B5",
    "82":"Bb5",
    "81":"A5",
    "80":"Ab5",
    "79":"G5",
    "78":"Gb5",
    "77":"F5",
    "76":"E5",
    "75":"Eb5",
    "74":"D5",
    "73":"Db5",
    "72":"C5",
    "71":"B4",
    "70":"Bb4",
    "69":"A4",
    "68":"Ab4",
    "67":"G4",
    "66":"Gb4",
    "65":"F4",
    "64":"E4",
    "63":"Eb4",
    "62":"D4",
    "61":"Db4",
    "60":"C4",
    "59":"B3",
    "58":"Bb3",
    "57":"A3",
    "56":"Ab3",
    "55":"G3",
    "54":"Gb3",
    "53":"F3",
    "52":"E3",
    "51":"Eb3",
    "50":"D3",
    "49":"Db3",
    "48":"C3",
    "47":"B2",
    "46":"Bb2",
    "45":"A2",
    "44":"Ab2",
    "43":"G2",
    "42":"Gb2",
    "41":"F2",
    "40":"E2",
    "39":"Eb2",
    "38":"D2",
    "37":"Db2",
    "36":"C2",
    "35":"B1",
    "34":"Bb1",
    "33":"A1",
    "32":"Ab1",
    "31":"G1",
    "30":"Gb1",
    "29":"F1",
    "28":"E1",
    "27":"Eb1",
    "26":"D1",
    "25":"Db1",
    "24":"C1",
    "23":"B0",
    "22":"Bb0",
    "21":"A0"
}

def midi_to_note(note_number):
    return midi_to_note_dict[str(note_number)]

def tokens_to_notes(tokens):
    return "".join([midi_to_note(int(token[1:]))[0] for token in tokens])

def tokens_to_notes_full(tokens):
    return " ".join([midi_to_note(int(token[1:])) for token in tokens])

def mutation_swap(original_perm, all_tokens=None):
    new_perm = list(original_perm)
    x = range(0,len(new_perm))
    shuffle(x)
    i = x[0] # random.randint(0,len(new_perm) - 1)
    j = x[1] # random.randint(0,len(new_perm) - 1)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def mutation_replace(original_perm, all_tokens):
    remaining = set(all_tokens).difference(original_perm)
    new_token = random.choice(list(remaining))
    new_perm = list(original_perm)
    i = random.randint(0,len(new_perm) - 1)
    new_perm[i] = new_token
    return new_perm

def test_mutation_replace():
    orig = [chr(x) for x in range(ord('a'),ord('z')+1)]
    n = 10
    partial = orig[0:n]
    for i in range(0,100000):
        new_perm = mutation_replace(partial, orig)
        assert partial != new_perm
        assert len(set(partial).difference(set(new_perm))) == 1

def test_mutation_swap():
    orig = [chr(x) for x in range(ord('a'),ord('z')+1)]
    n = 10
    partial = orig[0:n]
    for i in range(0,100000):
        new_perm = mutation_swap(partial, orig)
        assert partial != new_perm
        assert len(set(partial).difference(set(new_perm))) == 0
        assert set(partial) ==  set(new_perm)

def mutate_perm(original_perm, all_tokens, n_mutations=10):
    n_mutations = random.randint(1,n_mutations)
    mutaters = [mutation_swap, mutation_replace]
    new_perm = list(original_perm)
    for i in range(0, n_mutations):
        mutant_f = random.choice(mutaters)
        new_perm = mutant_f(new_perm, all_tokens)
    return new_perm
