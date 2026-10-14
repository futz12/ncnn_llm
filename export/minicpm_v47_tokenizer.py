"""Export exact MiniCPM tokenizer Unicode data using its tokenizers backend.

No Torch/Transformers dependency. Rejects unsupported tokenizer pipelines instead
of approximating a different regex or normalizer. The artifact is opt-in; legacy
BPE model behavior is unaffected.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import time

import tokenizers
from tokenizers import Regex, pre_tokenizers
from minicpm_v47_unicode import derive_nfc_tables, write_nfc_records

PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
MAGIC = 'minicpm_v47_tokenizer_v1'


def validate_pipeline(data):
    expected = {'type':'Sequence','pretokenizers':[
        {'type':'Split','pattern':{'Regex':PATTERN},'behavior':'Isolated','invert':False},
        {'type':'ByteLevel','add_prefix_space':False,'trim_offsets':False,'use_regex':False}]}
    if data.get('pre_tokenizer') != expected or data.get('normalizer') != {'type':'NFC'}:
        raise ValueError('unsupported MiniCPM pretokenizer regex/normalizer pipeline')
    byte_level = {'type':'ByteLevel','add_prefix_space':False,'trim_offsets':False,'use_regex':False}
    if data.get('decoder') != byte_level or data.get('post_processor') != byte_level:
        raise ValueError('unsupported MiniCPM decoder/postprocessor')
    if data.get('truncation') is not None or data.get('padding') is not None:
        raise ValueError('configured tokenizer truncation/padding is unsupported')
    model=data['model']
    if any('\n' in token or '\r' in token for token in model['vocab']):
        raise ValueError('raw newline vocabulary entries are unsupported by literal byte-BPE files')
    expected_model={'type':'BPE','dropout':None,'unk_token':None,'continuing_subword_prefix':'',
                    'end_of_word_suffix':'','fuse_unk':False,'byte_fallback':False,'ignore_merges':False}
    if {k:v for k,v in model.items() if k not in ('vocab','merges')} != expected_model:
        raise ValueError('unsupported MiniCPM BPE model options')
    for entry in data.get('added_tokens',[]):
        if any(entry.get(k) for k in ('single_word','lstrip','rstrip','normalized')):
            raise ValueError('unsupported added-token matching/normalization flags')
        if any(ord(c)<33 or ord(c)>126 for c in entry.get('content','')):
            raise ValueError('only checkpoint ASCII added-token literals are supported')
        if not entry.get('content') or not isinstance(entry.get('id'),int):
            raise ValueError('invalid added token')


def classification_ranges(batch_size=4096):
    """Classify through the same Rust/Oniguruma engine, not Python Unicode data."""
    if not 1 <= batch_size <= 65536:
        raise ValueError('batch_size must be between 1 and 65536')
    patterns=[r'\p{L}+',r'\p{M}+',r'\p{N}+',r'\s+']
    patterns += ['(?i:'+letter+')+' for letter in 'strev mld'.replace(' ','')]
    # Flags 0..3 are L/M/N/whitespace; 4..11 are s,t,r,e,v,m,l,d folds.
    values=array('H',[0])*0x110000
    for bit,pattern in enumerate(patterns):
        split=pre_tokenizers.Split(Regex(pattern),behavior='removed',invert=True)
        for lower,upper in ((0,0xD800),(0xE000,0x110000)):
            for begin in range(lower,upper,batch_size):
                end=min(begin+batch_size,upper)
                text=''.join(map(chr,range(begin,end)))
                for _,(a,b) in split.pre_tokenize_str(text):
                    for cp in range(begin+a,begin+b): values[cp]|=1<<bit
    ranges=[]; start=0; current=values[0]
    for cp in range(1,len(values)+1):
        value=values[cp] if cp<len(values) else -1
        if value!=current:
            if current: ranges.append((start,cp-1,current))
            start=cp; current=value
    return ranges


def export_minicpm_v47_tokenizer(model_dir,output_dir,filename='minicpm_v47_tokenizer.txt'):
    source=Path(model_dir)/'tokenizer.json'
    raw=source.read_bytes(); data=json.loads(raw)
    validate_pipeline(data)
    started=time.monotonic()
    ranges=classification_ranges()
    tables=derive_nfc_tables()
    path=Path(output_dir)/filename; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(MAGIC+'\n')
        stream.write('regex '+PATTERN+'\n')
        stream.write('source '+hashlib.sha256(raw).hexdigest()+'\n')
        stream.write('backend '+tokenizers.__version__+'\n')
        for first,last,flags in ranges: stream.write(f'R {first} {last} {flags}\n')
        write_nfc_records(stream,tables)
        for item in data['added_tokens']:
            stream.write(f"A {item['id']} {int(item['special'])} {item['content'].encode().hex()}\n")
        stream.write('END\n')
    print(f'[tokenizer] {len(ranges)} backend-classification ranges; artifact {path.stat().st_size} bytes; {time.monotonic()-started:.2f}s',flush=True)
    return {'type':'minicpm_v47','file':filename,'source_sha256':hashlib.sha256(raw).hexdigest(),
            'tokenizers_version':tokenizers.__version__}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',required=True); ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    print(json.dumps(export_minicpm_v47_tokenizer(args.model_dir,args.output_dir),indent=2))
