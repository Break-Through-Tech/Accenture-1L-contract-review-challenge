"""Stage 5: automated, evidence-backed resolution of the eight risk flags.

Combines three sources of evidence per contract:
  1. Part A clause detections (category, probability, chunk text),
  2. sentence-level cues inside the detected chunks,
  3. a search of the full cleaned contract text.

Each flag resolves to present / absent / not_applicable when the evidence agrees,
and stays `unknown` (human review) when it conflicts or is missing. Every resolved
flag carries `reviewer = RULES_VERSION`, a rationale and, for present risks, an exact
quote from the contract so it passes risk_scoring.engine validation and stays
distinguishable from human-confirmed assessments.

Contract triage then auto-classifies a contract whenever the remaining unknown flags
cannot change its bucket (engine minimum_bucket == maximum_bucket) and escalates the rest.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from risk_scoring.engine import FLAGS, default_config, score_contracts

RULES_VERSION = 'auto:rules-v1'
SPLIT_FILES = {'train': 'train', 'validation': 'val', 'val': 'val', 'test': 'test'}
FLAGS_RE = re.IGNORECASE | re.DOTALL


def clean_contract_text(text):
    """Identical to Part A (milestone3_task2a_detections.ipynb) so offsets and quotes line up."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence-level text cues
# ---------------------------------------------------------------------------
_BOUNDARY = re.compile(r'(?<=[.;])\s+(?=[A-Z0-9("\-])|\n\s*\n')


def sentences(text):
    """Split into sentence-like spans; every returned string is an exact substring of text."""
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        piece = text[start:m.start()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _rx(*parts):
    return re.compile('|'.join(parts), FLAGS_RE)


LIABILITY = _rx(r'\bliab(?:le|ility|ilities)\b', r'\bdamages\b')
CAP_PARTS = tuple(rf'\b{p}\b' for p in (
    r'(?:shall|will|may)\s+(?:in\s+no\s+(?:event|circumstances?|case)\s+)?(?:not\s+)?exceed',
    r'not\s+to\s+exceed',
    r'(?:be\s+)?limited\s+to\s+(?:the\s+|a\s+|an\s+)?(?:amount|total|aggregate|sum|fees|price|value|actual|direct|us\$|one|two|three|\d+)',
    r'(?:maximum|aggregate|cumulative|total)\s+(?:aggregate\s+|cumulative\s+)?liability',
    r'up\s+to\s+a\s+maximum',
    r'cap\s+on\s+(?:its\s+)?liability',
    r'in\s+(?:an\s+)?amount\s+(?:greater\s+than|in\s+excess\s+of)',
    r'in\s+no\s+event\s+(?:shall|will)\b[^.;]{0,120}\bexceed',
    r'exceed(?:s)?\s+(?:the\s+)?(?:amount|total|aggregate|sum|fees|price|\$|us\$|\d)',
))
CAP = _rx(*CAP_PARTS)
# "liability" as a limitation subject (not "limited liability company" or "losses ... or liabilities").
_LIAB = r'(?<!limited\s)\bliab(?:le|ility)\b(?!\s+compan)'
CAP_NEAR_LIABILITY = _rx(*(f'{_LIAB}[^.;]{{0,150}}(?:{p})' for p in CAP_PARTS),
                         *(f'(?:{p})[^.;]{{0,60}}{_LIAB}' for p in CAP_PARTS))
_INDIRECT = r'(?:consequential|incidental|indirect|special|punitive|exemplary|lost\s+profits?|loss\s+of\s+profits?)'
DAMAGES_EXCLUSION = _rx(
    rf'\b(?:not|no|never|neither)\b[^.;]{{0,200}}\b(?:liab(?:le|ility)|responsible)\b[^.;]{{0,300}}\b{_INDIRECT}',
    rf'\bliab(?:le|ility)\b[^.;]{{0,60}}\bfor\s+(?:any\s+)?{_INDIRECT}',
    rf'\bwaives?\b[^.;]{{0,200}}\b{_INDIRECT}',
    rf'\b(?:excluding|exclusion\s+of)\b[^.;]{{0,60}}\b{_INDIRECT}',
    r'\bliability\s+exclusion\b',
    r'\bsole\s+(?:and\s+exclusive\s+)?(?:obligation|remedy|liability)\b',
    # time limit on bringing claims (CUAD counts this as a cap on liability)
    r'\b(?:no\s+(?:action|claim)|may\s+(?:not\s+)?(?:be\s+)?(?:bring|brought))\b[^.;]{0,200}\bmore\s+than\b[^.;]{0,40}\b(?:years?|months?)\b',
)
# A carve-out must refer back to a limitation/cap, not just say "including without limitation".
CAP_REFERENCE = _rx(
    r'\blimitations?\s+(?:of|on)\s+(?:liab\w*|damages)',
    r'\b(?:foregoing|above|preceding|such)\s+(?:limitations?|caps?|exclusions?)',
    r'\blimitations?\s+(?:set\s+forth\s+)?(?:in|under|of)\s+(?:this\s+)?(?:section|article|clause|paragraph)',
    r'\bliability\s+cap\b', r'\bexceed\b', r'\bcap\b',
)
CARVEOUT = _rx(
    r'\b(?:shall|will|does|do)\s+not\s+apply\s+(?:to|with\s+respect\s+to|in\s+the\s+case\s+of)\b',
    r'\b(?:shall\s+be|is|are)\s+inapplicable\b',
    r'\bshall\s+not\s+(?:be\s+)?(?:subject\s+to|limit)\b[^.;]{0,80}\b(?:limitation|cap|foregoing)',
    r'\bexclud\w+\s+from\s+(?:the\s+)?(?:foregoing\s+|above\s+)?(?:limitation|cap)',
    r'\b(?:except|excluding|other\s+than|save)\s+(?:for|in\s+(?:the\s+)?(?:case|connection)\s+(?:of|with)|with\s+(?:regard|respect)\s+to|as\s+to)?\b[^.;]{0,300}?'
    r'\b(?:sections?\s+\d|articles?\s+\d|indemnif\w*|gross\s+negligence|wil+ful|intentional|fraud|confidential\w*|infring\w*|intellectual\s+property)',
)
# The sentence must actually be limiting liability for the carve-out to matter.
LIMIT_CONTEXT = _rx(
    r'\bin\s+no\s+event\b[^.;]{0,400}\bliab',
    r'\b(?:neither|no)\s+party\b[^.;]{0,200}\bliab',
    r'\bliab(?:le|ility)\b[^.;]{0,400}\b(?:exceed|consequential|incidental|indirect|special\s+damages|lost\s+profits|limited\s+to)',
    r'\blimitations?\s+(?:of|on)\s+(?:liab\w*|damages)', r'\bliability\s+cap\b',
    r'\b(?:foregoing|above|preceding|such)\s+(?:limitations?|caps?|exclusions?)',
)
UNLIMITED = _rx(
    r'\b(?:unlimited|uncapped)\s+(?:liabilit\w*|indemnif\w*)',
    r'\bliabilit\w*[^.;]{0,120}\b(?:unlimited|uncapped|without\s+(?:any\s+)?limit\b)',
)
INDEMNITY = _rx(r'\bindemnif\w*', r'\bhold\s+(?:\w+\s+){0,4}harmless\b')
INDEMNITY_CAPPED = _rx(
    r'\bindemnif\w*[^.;]{0,250}\bsubject\s+to\b[^.;]{0,120}\b(?:limitation|cap|section|article)',
    r'\bindemnif\w*[^.;]{0,250}\b(?:shall\s+not\s+exceed|not\s+to\s+exceed|limited\s+to)\b',
)
_TERM_RIGHT = (r'\b(?:may|can|shall\s+have\s+the\s+right\s+to|has\s+the\s+right\s+to|have\s+the\s+right\s+to|'
               r'(?:is|are|shall\s+be)\s+entitled\s+to|reserves?\s+the\s+right\s+to|right\s+to)\s+(?:\w+\s+){0,4}?terminat\w*')
_CONVENIENCE = (r'(?:for\s+convenience|for\s+any\s+reason|for\s+no\s+reason|without\s+cause|at\s+any\s+time|'
                r'with\s+or\s+without\s+cause|in\s+its\s+(?:sole\s+)?discretion|for\s+any\s+or\s+no\s+reason)')
TERM_CONVENIENCE = _rx(
    _TERM_RIGHT + r'[^.;]{0,200}\b' + _CONVENIENCE,
    r'\b' + _CONVENIENCE + r'\b[^.;]{0,120}' + _TERM_RIGHT,
    # "X may terminate this Agreement upon 90 days' written notice" with no stated cause
    _TERM_RIGHT + r'\s+(?:this\s+|the\s+)?(?:agreement|contract|engagement|license)?[^.;]{0,60}?\b(?:upon|by|with|on|after|giving|providing|by\s+giving)\b[^.;]{0,50}\bnotice\b',
    r'\bterminated\s+by\s+(?:either|any)\s+party\b',
)
# Sentences about consequences of termination, or termination for cause / on a trigger event, are not convenience rights.
TERM_FOR_CAUSE_ONLY = _rx(r'\b(?:breach|default|insolven\w*|bankrupt\w*|cure|change\s+of\s+control|failure)\b', r'\bsurviv\w*',
                          r'\bupon\s+(?:the\s+)?(?:expiration\s+or\s+)?termination\b',
                          r'\b(?:after|following|in\s+the\s+event\s+of)\s+(?:the\s+)?(?:expiration\s+or\s+)?termination\b',
                          r'\bif\s+(?!it\s+so\b)', r'\bin\s+the\s+event\s+(?:of|that)\b', r'\bfollowing\s+the\s+occurrence\b')
# Party named immediately before the termination right (case-sensitive: defined party names are capitalised).
TERM_SUBJECT = re.compile(r'([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})\s+(?:may|shall\s+have\s+the\s+right\s+to|has\s+the\s+right\s+to|'
                          r'(?:is|shall\s+be)\s+entitled\s+to|reserves\s+the\s+right\s+to)\s+(?:\w+\s+){0,3}?terminat', re.DOTALL)
MUTUAL = _rx(r'\b(?:either|each|both|any)\s+part(?:y|ies)\b', r'\bthe\s+parties\s+may\b', r'\bmutual(?:ly)?\b')
AUTO_RENEW = _rx(
    r'(?<!not\s)\bautomatic(?:ally)?\s+(?:be\s+)?(?:renew|extend)\w*',
    r'\b(?:renew|extend)\w*\s+automatically\b',
    r'\b(?:shall|will)\s+(?:thereafter\s+)?(?:be\s+)?(?:renewed|extended|continue)\s+(?:for|thereafter\s+for)\s+(?:successive|additional|consecutive|further)\b',
    r'\bevergreen\b',
    r'(?<!not\s)\bautomatic\s+(?:successive\s+)?(?:renewal|extension)s?\b',
    r'\brenewed\s+tacitly\b|\btacit(?:ly)?\s+renew\w*',
    r'\b(?:agreement|term)\b[^.;]{0,120}\b(?:continue|remain)\s+(?:in\s+(?:full\s+force\s+and\s+)?effect\s+)?(?:indefinitely|from\s+year\s+to\s+year|(?:on\s+a\s+)?month[\s-]+to[\s-]+month|year[\s-]+to[\s-]+year)\b',
    r'\b(?:indefinitely|thereafter)\b[^.;]{0,60}\buntil\s+terminated\b',
    r'\bautomatically\s+continue\b',
)
RENEWAL_NOTICE = _rx(
    r'\bnotice\b[^.;]{0,200}\b(?:non-?renewal|not\s+to\s+renew|intent\w*\s+not|terminat\w*|elect\w*)',
    r'\b(?:non-?renewal|not\s+to\s+renew)\b',
    r'\bunless\b[^.;]{0,120}\bnotice\b',
    r'\b(?:\d+|\(\d+\)|thirty|sixty|ninety|one\s+hundred)\b[^.;]{0,20}\bdays?\W{0,3}\s*(?:prior\s+)?(?:written\s+)?notice\b',
)
RENEWAL_BY_AGREEMENT = _rx(
    r'\b(?:renew\w*|extend\w*|extension)\b[^.;]{0,200}\b(?:mutual(?:ly)?\s+(?:written\s+)?agree\w*|by\s+(?:written\s+)?agreement|upon\s+(?:the\s+)?(?:mutual\s+)?(?:written\s+)?agreement|at\s+(?:the\s+)?option\s+of|option\s+to\s+(?:renew|extend)|may\s+(?:elect|request)\s+to)',
    r'\b(?:mutual(?:ly)?\s+agree\w*|option)\b[^.;]{0,120}\b(?:renew|extend)\w*',
    r'\b(?:renewable|may\s+be\s+(?:renewed|extended))\b',
    r'\b(?:renew\w*|extend\w*|extension|continuance)\b[^.;]{0,200}\b(?:written\s+consent|by\s+amendment|specifically\s+approved|consent\s+of\s+both)',
)
NEGATED_JOINT = _rx(r'\bnot\s+(?:be\s+)?(?:construed|deemed|intended)\b', r'\bnothing\b[^.;]{0,80}\b(?:create|constitute|imply)', r'\bpartnership\b', r'\bjoint\s+venture\b')
IP_CONTEXT = _rx(r'\bintellectual\s+property\b', r'\bpatents?\b', r'\bcopyrights?\b', r'\bwork\s+product\b', r'\binventions?\b', r'\bsource\s+code\b')
IP_OWNERSHIP = _rx(
    r'\bhereby\s+(?:irrevocably\s+)?assigns?\b',
    r'\bshall\s+(?:be\s+and\s+)?(?:remain\s+|be\s+)?(?:the\s+)?(?:sole\s+(?:and\s+exclusive\s+)?|exclusive\s+)?(?:property|owner)\b',
    r'\bshall\s+(?:solely\s+|exclusively\s+)?own\b',
    r'\b(?:retains?|reserves?)\s+(?:all\s+)?(?:right,?\s+title|ownership|rights?\s+in)\b',
    r'\bowned\s+(?:solely\s+|exclusively\s+)?by\b',
    r'\b(?:sole\s+and\s+)?exclusive\s+owner\b',
    r'\btitle\s+to\b[^.;]{0,80}\b(?:remain|vest)',
)
JOINT_IP = _rx(r'\bjointly\s+own\w*', r'\bjoint(?:ly)?\s+owne\w*', r'\bco-?own\w*', r'\bowned\s+jointly\b', r'\bjoint\s+(?:intellectual\s+property|inventions?)\b')
JOINT_ALLOCATION = _rx(r'\beach\s+party\s+shall\s+(?:own|retain)\b', r'\brespective\s+(?:ownership|rights)\b', r'\bwithout\s+(?:the\s+)?(?:need\s+to\s+)?account\w*\b')


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------
class Evidence:
    """Indexes one contract's full text and Part A detections."""

    def __init__(self, contract, full_text, margin=0.05):
        self.contract = contract
        self.text = full_text
        self.sents = sentences(full_text)
        self.margin = margin
        self.by_cat = {}
        for d in contract.get('detections', []):
            self.by_cat.setdefault(d['category'], []).append(d)

    def detected(self, *categories, strong=False):
        hits = [d for c in categories for d in self.by_cat.get(c, [])]
        if strong:
            hits = [d for d in hits if d.get('probability', 0) >= d.get('threshold', 0) + self.margin]
        return hits

    def find(self, pattern, *, also=None, unless=None, within=None):
        """Sentences matching pattern (and `also`, not `unless`), from full text or given chunks."""
        pool = self.sents if within is None else [s for d in within for s in sentences(d['text'])]
        out = []
        for s in pool:
            if pattern.search(s) and (also is None or also.search(s)) and (unless is None or not unless.search(s)):
                if s not in out:
                    out.append(s)
        return out

    def describe(self, cats):
        hits = self.detected(*cats)
        if not hits:
            return f"no Part A detection for {'/'.join(cats)} across {self.contract.get('chunks_scanned', '?')} chunks"
        best = max(hits, key=lambda d: d.get('probability', 0))
        return f"{len(hits)} Part A detection(s) for {'/'.join(cats)} (max p={best.get('probability'):.2f})"


def _quote(sentence, limit=600):
    """Exact substring for evidence; trimmed to keep the report readable."""
    return sentence if len(sentence) <= limit else sentence[:limit]


def _a(status, rationale, *, evidence=None, scope=None, confidence='rule'):
    out = {'status': status, 'rationale': rationale, 'reviewer': RULES_VERSION, 'method': 'rules+detections', 'confidence': confidence}
    if evidence is not None:
        out['evidence'] = _quote(evidence)
    if scope is not None:
        out['scope'] = scope
    return out


def _unknown(rationale, candidates=()):
    out = {'status': 'unknown', 'rationale': rationale, 'reviewer': RULES_VERSION, 'method': 'rules+detections'}
    if candidates:
        out['evidence_candidates'] = [_quote(c, 400) for c in candidates[:3]]
    return out


# ---------------------------------------------------------------------------
# Per-flag rules
# ---------------------------------------------------------------------------
LIAB_CATS = ('Cap On Liability', 'Uncapped Liability')


def assess_liability(ev):
    caps = ev.find(CAP_NEAR_LIABILITY)
    exclusions = ev.find(DAMAGES_EXCLUSION)
    carve = ev.find(CARVEOUT, also=LIMIT_CONTEXT)
    unlimited = ev.find(UNLIMITED)
    uncapped_label = ev.detected('Uncapped Liability', strong=True)
    det = ev.describe(LIAB_CATS)
    out = {}

    # missing_liability: is there any limitation-of-liability protection at all?
    if caps or exclusions:
        out['missing_liability'] = _a('absent', f'Limitation-of-liability language found in full text; {det}.', evidence=(caps or exclusions)[0])
    elif not ev.detected(*LIAB_CATS):
        out['missing_liability'] = _a('present', f'Full-text search found no cap or damages-exclusion language and {det}.', scope='full_contract')
    else:
        out['missing_liability'] = _unknown(f'Model detected a liability clause but no cap/exclusion wording was found; {det}.',
                                            [s for d in ev.detected(*LIAB_CATS) for s in sentences(d['text']) if LIABILITY.search(s)])

    # uncapped_liability: is an existing cap carved out / liability explicitly unlimited?
    if out['missing_liability']['status'] == 'present':
        out['uncapped_liability'] = _a('not_applicable', 'No liability clause exists; exposure is captured by missing_liability.')
    elif unlimited or carve:
        src = (unlimited or carve)[0]
        out['uncapped_liability'] = _a('present', f'Liability is unlimited or carved out of the cap; {det}.', evidence=src,
                                       confidence='rule+model' if uncapped_label else 'rule')
    elif caps and not uncapped_label:
        out['uncapped_liability'] = _a('absent', f'Monetary cap found and no carve-out language; {det}.', evidence=caps[0])
    elif caps and uncapped_label:
        out['uncapped_liability'] = _unknown(f'Conflict: model labels Uncapped Liability but text shows a cap with no carve-out; {det}.', caps)
    elif exclusions and not uncapped_label:
        # CUAD "Uncapped Liability" means an explicit carve-out from a limitation; none exists here.
        out['uncapped_liability'] = _a('absent', f'Only damages-type exclusions, with no carve-out from them; note direct damages have no amount cap; {det}.',
                                       evidence=exclusions[0], confidence='rule-low')
    else:
        out['uncapped_liability'] = _unknown(f'Liability clause detected but neither a cap nor a carve-out could be confirmed; {det}.', exclusions)
    return out


def assess_indemnification(ev, liability):
    indem = ev.find(INDEMNITY)
    out = {}
    if not indem:
        out['missing_indemnification'] = _a('present', 'Full-text search found no indemnify / hold-harmless language '
                                            f'(indemnification is not a CUAD category, so this is text-only); {ev.contract.get("chunks_scanned", "?")} chunks scanned.',
                                            scope='full_contract')
        out['open_ended_indemnification'] = _a('not_applicable', 'No indemnification clause; covered by missing_indemnification.')
        return out
    out['missing_indemnification'] = _a('absent', 'Indemnification language found in full text.', evidence=indem[0])

    excluded = [s for s in ev.find(CARVEOUT, also=INDEMNITY) if CAP_REFERENCE.search(s)]
    unlimited = ev.find(UNLIMITED, also=INDEMNITY)
    capped = ev.find(INDEMNITY_CAPPED) or ev.find(CAP, also=INDEMNITY, unless=CARVEOUT)
    if excluded or unlimited:
        out['open_ended_indemnification'] = _a('present', 'Indemnity obligations are excluded from the liability cap or stated as unlimited.',
                                               evidence=(unlimited or excluded)[0])
    elif capped:
        out['open_ended_indemnification'] = _a('absent', 'Indemnity is expressly subject to a cap or limitation.', evidence=capped[0])
    elif liability['uncapped_liability']['status'] == 'absent' and liability['uncapped_liability'].get('confidence') != 'rule-low':
        out['open_ended_indemnification'] = _a('absent', 'A general liability cap with no carve-outs applies to all liability, including indemnity.',
                                               evidence=liability['uncapped_liability']['evidence'])
    elif liability['missing_liability']['status'] == 'present':
        out['open_ended_indemnification'] = _a('present', 'Indemnity present and the contract has no limitation of liability, so indemnity is uncapped.',
                                               evidence=indem[0])
    else:
        out['open_ended_indemnification'] = _unknown('Indemnity present; could not tell whether the general cap applies to it.', indem)
    return out


def assess_termination(ev):
    det = ev.describe(('Termination For Convenience',))
    hits = ev.find(TERM_CONVENIENCE, unless=TERM_FOR_CAUSE_ONLY)
    mutual = [s for s in hits if MUTUAL.search(s)]
    one_way = [s for s in hits if not MUTUAL.search(s)]
    subjects = {m.group(1).split()[-1].strip('.').lower() for s in one_way for m in TERM_SUBJECT.finditer(s)}
    subjects -= {'this', 'the', 'agreement', 'notwithstanding', 'and', 'or'}
    if one_way and not mutual and len(subjects) >= 2:
        return _a('absent', f'Each party holds its own termination-for-convenience right ({", ".join(sorted(subjects))}); {det}.', evidence=one_way[0])
    if one_way and not mutual:
        return _a('present', f'Termination-for-convenience right granted to one party only; {det}.', evidence=one_way[0])
    if mutual and not one_way:
        return _a('absent', f'Termination-for-convenience right is mutual; {det}.', evidence=mutual[0])
    if mutual and one_way:
        return _unknown(f'Both mutual and one-way termination rights found; party roles need checking; {det}.', one_way + mutual)
    if not ev.detected('Termination For Convenience', strong=True):
        return _a('absent', f'No termination-for-convenience wording in full text and {det}.')
    return _unknown(f'Model detected a termination-for-convenience clause but wording did not match rules; {det}.',
                    [s for d in ev.detected('Termination For Convenience') for s in sentences(d['text']) if re.search(r'terminat', s, re.I)])


def assess_renewal(ev):
    cats = ('Renewal Term', 'Notice Period To Terminate Renewal')
    det = ev.describe(cats)
    auto = ev.find(AUTO_RENEW)
    if not auto:
        by_agreement = ev.find(RENEWAL_BY_AGREEMENT)
        if by_agreement:
            return _a('absent', f'Renewal requires an affirmative election or mutual agreement, not automatic; {det}.', evidence=by_agreement[0])
        if ev.detected('Renewal Term', strong=True):
            return _unknown(f'Model detected a renewal term but no automatic-renewal wording was found; {det}.',
                            [s for d in ev.detected('Renewal Term') for s in sentences(d['text']) if re.search(r'renew|extend', s, re.I)])
        return _a('not_applicable', f'No automatic renewal in full text; {det}.')
    with_notice = [s for s in auto if RENEWAL_NOTICE.search(s)]
    if with_notice:
        return _a('absent', f'Automatic renewal includes a notice / opt-out mechanism; {det}.', evidence=with_notice[0])
    notice_elsewhere = ev.find(RENEWAL_NOTICE, also=_rx(r'\brenew\w*', r'\bterm\b'))
    if notice_elsewhere or ev.detected('Notice Period To Terminate Renewal', strong=True):
        return _a('absent', f'Automatic renewal with a separate notice-to-terminate provision; {det}.',
                  evidence=(notice_elsewhere or auto)[0])
    return _a('present', f'Contract renews automatically with no notice / opt-out wording found; {det}.', evidence=auto[0])


def assess_ip(ev):
    cats = ('Ip Ownership Assignment', 'Joint Ip Ownership', 'License Grant')
    det = ev.describe(cats)
    relevant = ev.detected('License Grant', 'Ip Ownership Assignment', 'Joint Ip Ownership', strong=True) \
        or len(re.findall(r'intellectual\s+property', ev.text, re.I)) >= 2 or bool(ev.find(IP_OWNERSHIP, also=IP_CONTEXT))
    if not relevant:
        na = _a('not_applicable', f'Contract does not deal with intellectual property in substance; {det}.')
        return {'missing_ip_ownership': na, 'ambiguous_ip_ownership': dict(na)}
    owners = ev.find(IP_OWNERSHIP, also=IP_CONTEXT)
    joint = ev.find(JOINT_IP, unless=NEGATED_JOINT)
    out = {}
    if owners or joint:
        out['missing_ip_ownership'] = _a('absent', f'IP ownership language found; {det}.', evidence=(owners or joint)[0])
    elif not ev.detected('Ip Ownership Assignment', 'Joint Ip Ownership'):
        out['missing_ip_ownership'] = _a('present', f'IP is relevant but no ownership/assignment language found in full text and {det}.', scope='full_contract')
    else:
        out['missing_ip_ownership'] = _unknown(f'Model detected IP ownership clause but no ownership wording matched; {det}.')

    if out['missing_ip_ownership']['status'] == 'present':
        out['ambiguous_ip_ownership'] = _a('not_applicable', 'No IP ownership clause; covered by missing_ip_ownership.')
    elif joint and not any(JOINT_ALLOCATION.search(s) for s in joint):
        out['ambiguous_ip_ownership'] = _a('present', 'IP is jointly owned with no allocation of rights between the parties.', evidence=joint[0])
    elif owners and not joint:
        out['ambiguous_ip_ownership'] = _a('absent', f'Single-owner assignment / retention language; {det}.', evidence=owners[0])
    else:
        out['ambiguous_ip_ownership'] = _unknown(f'Joint or unclear ownership signals need review; {det}.', joint or owners)
    return out


def assess_contract(contract, full_text, margin=0.05):
    ev = Evidence(contract, full_text, margin)
    a = assess_liability(ev)
    a.update(assess_indemnification(ev, a))
    a['one_sided_termination'] = assess_termination(ev)
    a['renewal_without_notice'] = assess_renewal(ev)
    a.update(assess_ip(ev))
    return {f: a[f] for f in FLAGS}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def load_contract_texts(split, repo=Path('.')):
    df = pd.read_csv(repo / 'notebooks' / f'{SPLIT_FILES[split]}_contracts.csv')
    return dict(zip(df['contract_title'], df['context'].fillna('').map(clean_contract_text)))


def build_records(payload, texts, margin=0.05):
    if payload.get('smoke_test') is not False:
        raise ValueError('Refusing smoke-test or unverified input')
    records = []
    for contract in payload['contracts']:
        cid = contract['contract_id']
        if cid not in texts:
            raise ValueError(f'No full text for {cid}')
        records.append({
            'record_id': f"{contract['split']}:{cid}", 'contract_id': cid,
            'category': 'Full-contract risk triage', 'text': texts[cid],
            'source': f"notebooks/{SPLIT_FILES[contract['split']]}_contracts.csv (cleaned) + Part A detections",
            'split': contract['split'],
            'assessments': assess_contract(contract, texts[cid], margin),
        })
    return records


def triage(scored, records):
    """Auto-classify when unknown flags cannot change the bucket; otherwise escalate."""
    by_id = {r['record_id']: r for r in records}
    rows = []
    for s in scored:
        rec = by_id[s['record_id']]
        decided = s['bucket'] or (s['minimum_bucket'] if s['minimum_bucket'] == s['maximum_bucket'] else None)
        if not s['unknown_flags']:
            status = 'auto_classified'
        elif decided:
            status = 'auto_classified_unknowns_immaterial'
        else:
            status = 'human_review'
        findings = [{'flag': f, 'status': a['status'], 'why': a['rationale'], 'evidence': a.get('evidence'),
                     'evidence_candidates': a.get('evidence_candidates')}
                    for f, a in rec['assessments'].items() if a['status'] in ('present', 'unknown')]
        rows.append({
            'contract_id': s['contract_id'], 'split': s['split'], 'status': status,
            'risk_level': decided, 'score': s['score'], 'score_lower': s['score_lower'], 'score_upper': s['score_upper'],
            'possible_range': [s['minimum_bucket'], s['maximum_bucket']],
            'critical_overrides': s['critical_overrides'], 'unknown_flags': s['unknown_flags'],
            'review_reason': None if decided else
                f"Unresolved {', '.join(s['unknown_flags'])} could move risk between {s['minimum_bucket']} and {s['maximum_bucket']}",
            'findings': findings,
            'reviewer': RULES_VERSION,
        })
    return rows


def summarize(rows, records):
    status = Counter(r['status'] for r in rows)
    level = Counter(r['risk_level'] or 'needs_review' for r in rows)
    flags = {f: dict(Counter(rec['assessments'][f]['status'] for rec in records)) for f in FLAGS}
    return {'contracts': len(rows), 'status': dict(status), 'risk_level': dict(level), 'flag_status': flags}


def run(detections_path, out_dir, margin=0.05, config=None, repo=Path('.')):
    payload = json.loads(Path(detections_path).read_text(encoding='utf-8'))
    splits = {c['split'] for c in payload['contracts']}
    texts = {}
    for split in splits:
        texts.update(load_contract_texts(split, repo))
    records = build_records(payload, texts, margin)
    scored, _ = score_contracts(records, config or default_config())
    rows = triage(scored, records)
    summary = summarize(rows, records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = '_'.join(sorted(splits))
    (out / f'auto_assessments_{tag}.json').write_text(json.dumps(
        [{k: v for k, v in r.items() if k != 'text'} for r in records], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / f'auto_triage_{tag}.json').write_text(json.dumps({'rules_version': RULES_VERSION, 'summary': summary, 'contracts': rows},
                                                            indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description='Automated stage-5 risk flag assessment and triage')
    p.add_argument('--detections', default='outputs/pipeline/detections_test.json')
    p.add_argument('--out-dir', default='outputs/pipeline')
    p.add_argument('--margin', type=float, default=0.05, help='probability margin above threshold for a "strong" detection')
    args = p.parse_args()
    print(json.dumps(run(args.detections, args.out_dir, args.margin), indent=2))


if __name__ == '__main__':
    main()
