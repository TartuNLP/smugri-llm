# Detokenization with some normalization rules from Moses
import re

REGEXES_EST = (
    # parenthesis
    (re.compile(r' *\( *'), r' ('),
    (re.compile(r' *\) *'), r') '),
    (re.compile(r'\) ([.!:?;,])'), r')\1'),

    # whitespace character normalization
    (re.compile(r' '), ' '),
    (re.compile(r'\r'), r''),

    # remove unnnecessary spaces
    (re.compile(r'(\d) %'), r'\1%'),
    (re.compile(r' ([:;?!,.])'), r'\1'),

    # normalize quotes (ET test/dev sets have only " )
    (re.compile(r'[„“”«»《》]'), r'"'),

    (re.compile(r'"\s*([^"]*?)\s*"'), r'"\1"'),

    # remove extra spaces
    (re.compile(r' +'), r' '),
)

REGEXES_DEU = (
    # parenthesis
    (re.compile(r' *\( *'), r' ('),
    (re.compile(r' *\) *'), r') '),
    (re.compile(r'\) ([.!:?;,])'), r')\1'),

    # whitespace character normalization
    (re.compile(r' '), ' '),
    (re.compile(r'\r'), r''),

    # remove unnnecessary spaces
    (re.compile(r'(\d) %'), r'\1%'),
    (re.compile(r' ([:;?!,.])'), r'\1'),

    # normalize quotes
    (re.compile(r'[“«»《》]'), r'"'),

    (re.compile(r'"\s*([^"]*?)\s*"'), r'"\1"'),
    (re.compile(r'„\s*([^"]*?)\s*”'), r'„\1”'),
    (re.compile(r'„\s*([^"]*?)\s*"'), r'„\1"'),
    (re.compile(r'„\s+'), r'„'),
    (re.compile(r'\s+”'), r'”'),


    # remove extra spaces
    (re.compile(r' +'), r' '),
)


def detokenize_deu(sentence: str) -> str:
    for regex, sub in REGEXES_DEU:
        sentence = regex.sub(sub, sentence)

    return sentence


def detokenize_est(sentence: str) -> str:
    for regex, sub in REGEXES_EST:
        sentence = regex.sub(sub, sentence)

    return sentence