import pynini as pn
import pywrapfst as fst
import sys

sys.setrecursionlimit(20000)

import random

def build_marked_linear_fst(sentence, isyms):
  f = pn.Fst()
  one = fst.Weight.one(f.weight_type())

  cur = f.add_state()
  f.set_start(cur)

  for c in sentence:
    label = isyms.find(c)
    next = f.add_state()
    f.add_arc(cur, fst.Arc(label, label, one, next))
    cur = next

  f.set_final(cur)
  return f

'''
build a trie then optimize it to make it a nice acyclic DAG unweighted
character-to-word lexicon transducer.

see this paper: https://aclanthology.org/W19-3105.pdf
(though actually it seems there is a mistake in it, b:booted and b:epsilon
shouldn't both exist at the start state... whoops)

the optimizations are label pushing and then some minimizations
and determinizations (idk how pynini implements .optimize)

label pushing tends to make the lexicon a bit smaller, and have the
non-epsilon output labels evenly distributed throughout the word lengths
without it, the all tend to be at the very end of the words

we probably want to add a space character in here, as well as a __ token, then
we will do fst closure with the space key, so that it accepts anything in
(L ∪ SPACE)*
'''

MARK = "﹏"

def create_internal_marker_lexicon_old(words, isyms=None, osyms=None, start_marker = MARK):
  words = sorted(set(words))
  if not isyms:
    letters = sorted(set(c for w in words for c in w))
    isyms = pn.SymbolTable()
    isyms.add_symbol("<eps>", 0)
    for l in letters:
      isyms.add_symbol(l)

  if not osyms:
    osyms = pn.SymbolTable()
    osyms.add_symbol("<eps>", 0)
    for w in words:
      osyms.add_symbol(w)

  marker_label = isyms.add_symbol(start_marker)
  f = pn.Fst()
  one = fst.Weight.one(f.weight_type())

  start = f.add_state()
  f.set_start(start)
  
  marker_state = f.add_state()
  f.add_arc(start, fst.Arc(marker_label, 0, one, marker_state))


  for w in words:
    if w[0] == start_marker:
      cur = marker_state
      word = w[1:]
    else:
      cur = start
      word = w

    for (idx, c) in enumerate(word):

      next = f.add_state()
      if idx == len(word) - 1:
        f.add_arc(cur, fst.Arc(isyms.find(c), osyms.find(''.join(w)), one, next))
      else:
        f.add_arc(cur, fst.Arc(isyms.find(c), 0, one, next))
      cur = next
    f.set_final(cur, one)
    f.add_arc(cur, fst.Arc(0, 0, one, marker_state))

  f.set_input_symbols(isyms)
  f.set_output_symbols(osyms)

  '''
  https://github.com/kylebgorman/pynini/blob/0696c5087d36272dc41c20f9e6ebe351de5f67e5/pynini/__init__.pyi#L188
  https://www.openfst.org/twiki/pub/FST/FstSltTutorial/part1.pdf (<- slide 39)
  '''
  optimized = pn.push(f.optimize(), 1.0, False, True).optimize()
  optimized.arcsort()

  return isyms, osyms, optimized


MARK = '﹏'
def create_internal_marker_lexicon(words, isyms=None, osyms=None, start_marker = MARK):
  words = sorted(set(words))
  if not isyms:
    letters = sorted(set(c for w in words for c in w))
    isyms = pn.SymbolTable()
    isyms.add_symbol("<eps>", 0)
    space_sym = isyms.add_symbol('空')
    for l in letters:
      isyms.add_symbol(l)

  if not osyms:
    osyms = pn.SymbolTable()
    osyms.add_symbol("<eps>", 0)
    end_sym = osyms.add_symbol("끝")
    for w in words:
      osyms.add_symbol(w)

  marker_label = isyms.add_symbol(start_marker)
  f = pn.Fst()
  one = fst.Weight.one(f.weight_type())

  start = f.add_state()
  f.set_start(start)
  
  marker_state = f.add_state()
  f.add_arc(start, fst.Arc(marker_label, 0, one, marker_state))


  for w in words:
    if w[0] == start_marker:
      cur = marker_state
      word = w[1:]
    else:
      cur = start
      word = w

    for (idx, c) in enumerate(word):

      next = f.add_state()
      if idx == len(word) - 1:
        f.add_arc(cur, fst.Arc(isyms.find(c), osyms.find(''.join(w)), one, next))
      else:
        f.add_arc(cur, fst.Arc(isyms.find(c), 0, one, next))
      cur = next
    f.set_final(cur, one)
    f.add_arc(cur, fst.Arc(0, 0, one, marker_state))
    f.add_arc(cur, fst.Arc(space_sym, end_sym, one, start))

  f.set_input_symbols(isyms)
  f.set_output_symbols(osyms)

  '''
  https://github.com/kylebgorman/pynini/blob/0696c5087d36272dc41c20f9e6ebe351de5f67e5/pynini/__init__.pyi#L188
  https://www.openfst.org/twiki/pub/FST/FstSltTutorial/part1.pdf (<- slide 39)
  '''
  optimized = pn.push(f.optimize(), 1.0, False, True).optimize()

  return isyms, osyms, optimized


def _biased_dag_sampler(lattice, syms):
  cur = lattice.start()
  p = 1.0
  path = []
  while lattice.num_arcs(cur) > 0:
    index = random.choice(range(lattice.num_arcs(cur)))
    p *= (1.0 / lattice.num_arcs(cur))
    arc = list(lattice.arcs(cur))[index]
    if arc.olabel != 0:
      path.append(syms.find(arc.olabel))
    cur = arc.nextstate
  return path, p

def sample_from_lattice(word, lexicon, isyms, osyms):
  t = build_marked_linear_fst(word, isyms)
  # lattice = (t @ lexicon).project("output").rmepsilon()
  lattice = fst.compose(t, lexicon)
  if lattice.num_states() == 0: return ''
  p_min = 1.0
  for s in range(lattice.num_states()):
    if lattice.num_arcs(s) > 0:
      p_min *= (1.0 / lattice.num_arcs(s))
  
  path, p = _biased_dag_sampler(lattice, osyms)
  while random.random() > p_min / p:
    path, p = _biased_dag_sampler(lattice, osyms)
  
  return ' '.join(path)

def faster_dag_sampler(word, lexicon, isyms, osyms):
  global LINEAR, COMPOSITION, SAMPLE
  t = build_marked_linear_fst(word, isyms)
  # lattice = (t @ lexicon).project("output").rmepsilon()
  lattice = (t @ lexicon).project("output").rmepsilon()

  if lattice.num_states() == 0: return ''
  cache = {}

  def recur(state):
    if lattice.num_arcs(state) == 0: 
      cache[state] = 1
      return 1
    if state not in cache:
      cache[state] = sum(recur(a.nextstate) for a in lattice.arcs(state))
    return cache[state]
  recur(lattice.start())

  path_index = random.randint(0, cache[lattice.start()] - 1)
  path = []
  cur = lattice.start()
  while lattice.num_arcs(cur) > 0:

    running_sum = 0

    arcs = list(lattice.arcs(cur))
    if len(arcs) == 1:
      path.append(osyms.find(arcs[0].olabel))
      cur = arcs[0].nextstate
    else:
      next_arc = None
      for (i, arc) in enumerate(arcs):
        if running_sum <= path_index:
          running_sum += cache[arc.nextstate]
        else:
          next_arc = arcs[max(i-1, 0)]
      if not next_arc:
        next_arc = arcs[-1]
      if next_arc.olabel != 0:
        path.append(osyms.find(next_arc.olabel))
      cur = next_arc.nextstate
      path_index -= (running_sum - cache[arc.nextstate])
  # return path
  return ' '.join(path)

def faster_dag_sampler_no_proj(word, lexicon, isyms, osyms):
  t = build_marked_linear_fst(word, isyms)
  lattice = fst.compose(t, lexicon)

  if lattice.num_states() == 0: return ''
  cache = {}


  def recur(state):
    if lattice.num_arcs(state) == 0: 
      cache[state] = 1
      return 1
    if state not in cache:
      cache[state] = sum(recur(a.nextstate) for a in lattice.arcs(state))
    return cache[state]
  recur(lattice.start())


  path_index = random.randint(0, cache[lattice.start()] - 1)
  path = []
  cur = lattice.start()
  while lattice.num_arcs(cur) > 0:

    running_sum = 0
    arcs = list(lattice.arcs(cur))
    if len(arcs) == 1:
      if arcs[0].olabel != 0:
        path.append(osyms.find(arcs[0].olabel))
      cur = arcs[0].nextstate
    else:
      next_arc = None
      running_sum = 0
      for (i, arc) in enumerate(arcs):

        if running_sum + cache[arc.nextstate] <= path_index:
          running_sum += cache[arc.nextstate]
        else:
          next_arc = arcs[i]
          break
      if not next_arc:
        next_arc = arcs[-1]
      if next_arc.olabel != 0:
        path.append(osyms.find(next_arc.olabel))
      cur = next_arc.nextstate
      path_index -= running_sum

  return ' '.join(path)