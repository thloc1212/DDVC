"""Training-only word-concept vocabulary; no POS model or validation captions."""
import collections
import hashlib
import json
from pathlib import Path
import re

# These are content-word concepts, not externally annotated action/object classes.
STOP_WORDS = frozenset('''a an the and or but if as at by for from in into of on onto
to with without is am are was were be been being do does did done have has had
having can could may might must shall should will would it its itself this that
these those there here then than so very just now next also again while when
where which who whom whose what why how i me my we us our you your he him his
she her they them their some any each every both all another other one two
three four five first second third before after up down out off over under
through around back about across along during until toward towards s t d ll re ve
'''.split())


def concept_words(sentence):
    return {word for word in re.findall(r"[a-z]+", sentence.lower())
            if len(word) > 2 and word not in STOP_WORDS}


def make_vocabulary(records, max_concepts=512, min_frequency=5):
    counts = collections.Counter()
    total = 0
    for record in records.values():
        for sentence in record['sentences']:
            counts.update(concept_words(sentence)); total += 1
    concepts = sorted((word for word, count in counts.items() if count >= min_frequency),
                      key=lambda word: (-counts[word], word))[:max_concepts]
    if not concepts:
        raise ValueError('No training concepts pass min_frequency; inspect training captions')
    return dict(version=1, tokenizer='lower_ascii_content_words_v1', concepts=concepts,
                document_counts=[counts[word] for word in concepts], caption_count=total,
                max_concepts=max_concepts, min_frequency=min_frequency,
                label_source='training event captions only; no validation/test captions or POS model')


def load_vocabulary(path, expected_sha256=''):
    payload = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Event-concept vocabulary SHA256 mismatch')
    vocab = json.loads(payload)
    words, counts, total = vocab['concepts'], vocab['document_counts'], vocab['caption_count']
    if vocab.get('version') != 1 or vocab.get('tokenizer') != 'lower_ascii_content_words_v1':
        raise ValueError('Unsupported concept vocabulary/tokenizer')
    if not words or len(words) != len(set(words)) or len(words) != len(counts):
        raise ValueError('Invalid event-concept vocabulary')
    if total < 1 or any(not isinstance(n, int) or n < 1 or n > total for n in counts):
        raise ValueError('Invalid concept document frequencies')
    return vocab
