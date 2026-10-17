#!/usr/bin/env python3
"""100-example LoCoMo/LongMemEval memory benchmark (gpt-6-luna, Responses API).

Experiment 1: per-turn vs 5-turn extraction, using identical add-only storage.
Select winner on 20 development QAs (10 per corpus).
Experiment 2: four memory update policies, evaluate on 80 held-out QAs.
Total unique QA examples = 100 (50 per corpus); no fabricated results.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any
from urllib.request import urlretrieve

DATA_URLS = {
    'locomo': 'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json',
    'longmemeval': 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json',
}


def jdump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(obj):
    return hashlib.sha256(jdump(obj).encode('utf-8')).hexdigest()


def tokenize(s):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', str(s)).casefold())


def normalize(s):
    return ' '.join(tokenize(s))


def bm25_rank(query, mems, k=8):
    """Simple dependency-free BM25 for reproducible lexical memory search."""
    if not mems: return []
    q = set(tokenize(query))
    if not q: return sorted(mems, key=lambda x: x['seq'], reverse=True)[:k]
    doc_tokens = [tokenize(m['text']) for m in mems]
    df = collections.Counter(t for ts in doc_tokens for t in set(ts))
    avglen = max(1., sum(map(len, doc_tokens)) / len(doc_tokens))
    n = len(doc_tokens)
    scored = []
    for i, ts in enumerate(doc_tokens):
        tf = collections.Counter(ts); score = 0.
        for term in q:
            if term not in tf: continue
            idf = math.log(1. + (n - df[term] + .5) / (df[term] + .5))
            score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (.25 + .75 * len(ts) / avglen))
        scored.append((score, mems[i]['seq'], i))
    scored.sort(reverse=True)
    return [mems[i] for _, _, i in scored[:k]]


def chunks(turns, batch):
    """Non-overlapping, session-bound chunks, chronological order preserved."""
    out = []
    cur = []
    for turn in turns:
        if cur and (len(cur) >= batch or cur[-1]['session'] != turn['session']):
            out.append(cur); cur = []
        cur.append(turn)
    if cur: out.append(cur)
    return out


def build_locomo(data):
    cases = []
    for convo in data:
        ident = str(convo.get('sample_id', len(cases)))
        source = convo['conversation']
        sessions = sorted((int(m.group(1)), key) for key in source for m in [re.fullmatch(r'session_(\d+)', key)] if m)
        turns = []
        for number, key in sessions:
            dt = str(source.get(key + '_date_time', ''))
            for j, turn in enumerate(source[key]):
                content = str(turn.get('text', '')).strip()
                if turn.get('blip_caption'): content += ' [image caption: ' + str(turn['blip_caption']) + ']'
                if not content: continue
                turns.append({'id': str(turn.get('dia_id', f'{key}:{j}')), 'session': key,
                              'date': dt, 'speaker': str(turn.get('speaker', 'speaker')),
                              'text': content})
        for i, qa in enumerate(convo.get('qa', [])):
            if not qa.get('answer') or not qa.get('question'): continue
            cases.append({'id': f'locomo:{ident}:{i}', 'group': f'locomo:{ident}',
                          'dataset': 'locomo', 'type': str(qa.get('category', 'unknown')),
                          'question': str(qa['question']), 'answer': str(qa['answer']),
                          'evidence': [str(v) for v in qa.get('evidence', [])], 'turns': turns})
    return cases


def build_longmemeval(data):
    cases = []
    for i, row in enumerate(data):
        packed = list(zip(row['haystack_dates'], row['haystack_session_ids'], row['haystack_sessions']))
        packed.sort(key=lambda x: (str(x[0]), str(x[1])))
        turns, evidence = [], []
        for date, sid, session in packed:
            for j, t in enumerate(session):
                tid = f'{sid}:{j}'
                content = str(t.get('content', '')).strip()
                if not content: continue
                turns.append({'id': tid, 'session': str(sid), 'date': str(date),
                              'speaker': str(t.get('role', 'unknown')), 'text': content})
                if t.get('has_answer') is True: evidence.append(tid)
        qid = str(row.get('question_id', i))
        cases.append({'id': f'longmemeval:{qid}', 'group': f'longmemeval:{qid}',
                      'dataset': 'longmemeval', 'type': str(row.get('question_type', 'unknown')),
                      'question': str(row['question']), 'answer': str(row['answer']),
                      'evidence': evidence, 'turns': turns})
    return cases


def sample_cases(locomo, lm, seed):
    """Exactly 50 + 50; LoCoMo balanced across its 10 dialogues; LM over-samples updates."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for c in locomo: groups[c['group']].append(c)
    if len(groups) < 10: raise ValueError('LoCoMo must include all 10 conversations')
    loc_picked = []
    for g in sorted(groups):
        pool = groups[g]
        if len(pool) < 5: raise ValueError(f'Insufficient LoCoMo QAs in {g}')
        loc_picked += rng.sample(pool, 5)
    if len(loc_picked) != 50: raise ValueError('Expected exactly 50 LoCoMo items')

    lm_groups = collections.defaultdict(list)
    for c in lm: lm_groups[c['type']].append(c)
    # Prioritize update/temporal cases; then fill with other categories.
    quotas = {'knowledge-update': 20, 'temporal-reasoning': 10, 'multi-session': 10}
    lm_picked, chosen = [], set()
    for cat, n in quotas.items():
        pool = lm_groups.get(cat, [])
        use = rng.sample(pool, min(n, len(pool)))
        lm_picked.extend(use); chosen.update(c['id'] for c in use)
    rest = [c for c in lm if c['id'] not in chosen and c['type'] != 'knowledge-update']
    if len(rest) < 50 - len(lm_picked):
        rest += [c for c in lm if c['id'] not in chosen and c['type'] == 'knowledge-update']
    lm_picked += rng.sample(rest, 50 - len(lm_picked))
    if len(lm_picked) != 50: raise ValueError('Expected exactly 50 LongMemEval items')

    # Split 10 dev + 40 held-out in EACH dataset; report stratification.
    # Group-disjoint split for LoCoMo prevents dev/test sharing a base conversation.
    loc_groups = sorted({c['group'] for c in loc_picked})
    dev_groups = set(rng.sample(loc_groups, 2))
    rng.shuffle(lm_picked)
    dev = [c for c in loc_picked if c['group'] in dev_groups] + lm_picked[:10]
    test = [c for c in loc_picked if c['group'] not in dev_groups] + lm_picked[10:]
    return dev, test


# OpenAI Responses JSON Schemas. Full closed objects for strict structured outputs.
FACT_SCHEMA = {'type': 'object', 'properties': {'facts': {'type': 'array', 'items': {'type': 'object',
    'properties': {'text': {'type': 'string'}, 'source_turn_ids': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['text', 'source_turn_ids'], 'additionalProperties': False}}},
    'required': ['facts'], 'additionalProperties': False}
INDEX_SCHEMA = lambda fields: {'type': 'object', 'properties': {n: {'type': 'array', 'items': {'type': 'integer'}} for n in fields},
                               'required': list(fields), 'additionalProperties': False}
QUERY_SCHEMA = {'type': 'object', 'properties': {'query': {'type': 'string'}},
                'required': ['query'], 'additionalProperties': False}
ANSWER_SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}},
                 'required': ['answer'], 'additionalProperties': False}
JUDGE_SCHEMA = {'type': 'object', 'properties': {'correct': {'type': 'boolean'}},
                'required': ['correct'], 'additionalProperties': False}


class LLM:
    """Persistent content-addressed cache; logical call usage charged to each arm separately."""
    def __init__(self, model, root, price_in, price_out, offline=False):
        self.model = model; self.path = Path(root); self.path.mkdir(parents=True, exist_ok=True)
        self.price_in = price_in; self.price_out = price_out; self.offline = offline
        self.client = None
        if not offline:
            from openai import OpenAI
            self.client = OpenAI(max_retries=2, timeout=120.)
        self.actual_api_calls = 0

    def run(self, label, system, payload, schema, meter):
        key = sha({'model': self.model, 'label': label, 'system': system, 'payload': payload, 'schema': schema})
        path = self.path / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
        else:
            if self.offline: raise RuntimeError('Offline cache miss: ' + label + ' ' + key)
            delay = 1.
            for attempt in range(4):
                try:
                    start = time.perf_counter()
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[{'role':'system','content':system}, {'role':'user','content':jdump(payload)}],
                        text={'format':{'type':'json_schema','name':label,
                                        'strict':True,'schema':schema}},
                        store=False,
                    )
                    elapsed = time.perf_counter() - start
                    if not resp.output_text: raise ValueError('Empty or refused structured response: ' + label)
                    val = json.loads(resp.output_text)
                    usage = {'input': int(resp.usage.input_tokens if resp.usage else 0),
                             'output': int(resp.usage.output_tokens if resp.usage else 0),
                             'cached_input': int(getattr(getattr(resp.usage, 'input_tokens_details', None), 'cached_tokens', 0) or 0)}
                    saved = {'value': val, 'usage': usage, 'seconds': elapsed}
                    temp = path.with_suffix('.tmp')
                    temp.write_text(jdump(saved), encoding='utf-8')
                    temp.replace(path)
                    self.actual_api_calls += 1
                    break
                except Exception:
                    if attempt == 3:
                        raise RuntimeError(
                            f'API request failed after {attempt + 1} attempts '
                            f'(label={label}, cache_key={key})'
                        ) from None
                    time.sleep(delay); delay *= 2
        # All arms account for their own *logical* usage (even if common inputs are physically cached).
        meter['calls'] += 1
        meter['input_tokens'] += saved['usage']['input']
        meter['output_tokens'] += saved['usage']['output']
        meter['cached_input_tokens'] += saved['usage'].get('cached_input', 0)
        meter['latency_s'] += saved['seconds']
        return saved['value']


def new_meter():
    return {'calls': 0, 'cache_hits': 0, 'cache_misses': 0,
            'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0, 'latency_s': 0.}


def render_chunk(chunk):
    return [{'id': t['id'], 'date': t['date'], 'speaker': t['speaker'], 'text': t['text']} for t in chunk]


def extract(llm, chunk, meter):
    system = ("You extract durable, atomic, retrieval-useful, explicitly supported facts about named speakers/entities "
              "from dialogue. Each fact must be self-contained, attribute its subject, and preserve dates, change markers, "
              "negations, preferences, and time scope. Do not guess or invent. Do not store generic assistant advice or "
              "questions as user facts. A dialogue may have two human speakers. Return zero facts for trivial chatter. "
              "source_turn_ids must reference exact ids in the supplied turns supporting the fact. Keep previous historical "
              "states if explicitly historical; do not decide memory deletions here.")
    obj = llm.run('extract_facts', system, {'turns': render_chunk(chunk)}, FACT_SCHEMA, meter)
    ids = set(t['id'] for t in chunk)
    seen = set(); out = []
    for f in obj['facts']:
        t = str(f['text']).strip()
        if not t or normalize(t) in seen: continue
        seen.add(normalize(t))
        out.append({'text': t, 'source_ids': [str(v) for v in f['source_turn_ids'] if str(v) in ids],
                    'date': chunk[-1]['date']})
    return out


def rewrite(llm, chunk, fresh, meter):
    system = ("Generate one concise, entity-aware BM25 search query to find OLD memories about facts newly stated "
              "in the current dialogue. Resolve pronouns from this dialogue. Include names, concepts and alternative "
              "related terms as useful; do not introduce new facts. Only output query JSON.")
    obj = llm.run('memory_query', system, {'turns': render_chunk(chunk), 'fresh_facts': [f['text'] for f in fresh]},
                  QUERY_SCHEMA, meter)
    return str(obj['query']).strip() or ' '.join(f['text'] for f in fresh)


def indices(raw, field, upper, invalid_counter):
    vals = raw.get(field, []); good = set()
    for v in vals:
        if type(v) is not int or not (0 <= v < upper):
            invalid_counter[0] += 1
        else: good.add(v)
    return good


def update(llm, policy, mems, fresh, chunk, meter, topk=12):
    """All strategies receive IDENTICAL fresh extraction; retrieval is from pre-update active state."""
    if not fresh: return {'added':0, 'deleted':0, 'skipped':0, 'invalid_indices':0,
                          'trace': {'fresh':[], 'retrieved':[], 'skip_new_indices':[], 'delete_old_indices':[]}}
    invalid = [0]
    prior = []
    if policy != 'add_only':
        q = rewrite(llm, chunk, fresh, meter)
        prior = bm25_rank(q, mems, topk)
    fresh_text = [{'index': i, 'text': f['text']} for i, f in enumerate(fresh)]
    prior_text = [{'index': i, 'memory_id': m['id'], 'text': m['text'], 'date': m['date']}
                  for i, m in enumerate(prior)]
    payload = {'dialogue': render_chunk(chunk), 'fresh': fresh_text, 'retrieved': prior_text}
    common = ("Compare NEW facts with OLD RETRIEVED memories using the dialogue as primary evidence. "
              "Indices are ZERO-BASED and refer ONLY to the supplied indexed arrays. "
              "Delete OLD memories only when directly retracted, corrected, or superseded for the SAME subject and "
              "time scope. Do not delete distinct entities, historical facts, or facts that can both remain true. "
              "When there is uncertainty prefer keeping the old memory. Never delete an old memory solely because of "
              "a new fact you decided to skip. Ignore old memories not in retrieved list. ")
    skip, delete = set(), set()
    if policy == 'joint':
        obj = llm.run('joint_indices', common + "Return skip_new_indices for redundant/unsupported NEW facts and "
                      "delete_old_indices for contradicted OLD memories, in ONE decision.",
                      payload, INDEX_SCHEMA(['skip_new_indices', 'delete_old_indices']), meter)
        skip = indices(obj, 'skip_new_indices', len(fresh), invalid)
        delete = indices(obj, 'delete_old_indices', len(prior), invalid)
    elif policy == 'sequential':
        s = llm.run('skip_indices', common + "First stage: return skip_new_indices for NEW facts that are redundant "
                    "with retrieved memories or not supported by dialogue. Do not decide deletions yet.",
                    payload, INDEX_SCHEMA(['skip_new_indices']), meter)
        skip = indices(s, 'skip_new_indices', len(fresh), invalid)
        kept = [p for i, p in enumerate(fresh_text) if i not in skip]
        # Deletion decision deliberately sees only retained fresh facts, original indices stay stable.
        dpayload = {**payload, 'fresh': kept}
        d = llm.run('delete_indices', common + "Second stage: return delete_old_indices of OLD memories that are "
                    "contradicted by the RETAINED new facts. Do not delete old facts merely because omitted.",
                    dpayload, INDEX_SCHEMA(['delete_old_indices']), meter)
        delete = indices(d, 'delete_old_indices', len(prior), invalid)
    elif policy == 'delete_only':
        d = llm.run('delete_only_indices', common + "Keep ALL extracted new facts untouched. Return only indices of "
                    "retrieved OLD memories that must be deleted/superseded.",
                    payload, INDEX_SCHEMA(['delete_old_indices']), meter)
        delete = indices(d, 'delete_old_indices', len(prior), invalid)
    elif policy != 'add_only':
        raise ValueError('Unknown policy: ' + policy)
    ids_to_remove = {prior[i]['id'] for i in delete}
    # Logical tombstones are equivalent to deletion for retrieval; old states retained in audit only.
    mems[:] = [m for m in mems if m['id'] not in ids_to_remove]
    before = len(mems)
    exact = {normalize(m['text']) for m in mems}
    for i, f in enumerate(fresh):
        if i in skip or normalize(f['text']) in exact: continue
        mems.append({'id': sha([chunk[-1]['id'], i, f['text']])[:16],
                     'text': f['text'], 'source_ids': f['source_ids'], 'date': f['date'],
                     'seq': max([m['seq'] for m in mems], default=0) + 1})
        exact.add(normalize(f['text']))
    return {'added':len(mems)-before, 'deleted': len(ids_to_remove),
            'skipped':len(skip), 'invalid_indices':invalid[0],
            'trace': {'fresh':fresh_text, 'retrieved':prior_text,
                      'skip_new_indices':sorted(skip),'delete_old_indices':sorted(delete)}}


def apply_add_only(mems, fresh, chunk):
    exact = {normalize(m['text']) for m in mems}
    before = len(mems)
    for i, fact in enumerate(fresh):
        if normalize(fact['text']) in exact: continue
        mems.append({'id':sha([chunk[-1]['id'],i,fact['text']])[:16],
                     'text':fact['text'],'source_ids':fact.get('source_ids',[]),
                     'date':fact['date'],'seq':max([m['seq'] for m in mems],default=0)+1})
        exact.add(normalize(fact['text']))
    return {'added':len(mems)-before,'deleted':0,'skipped':0,'invalid_indices':0}


def make_history(llm, case, batch, policy, meter, cache, topk):
    key = (sha(case['turns']), batch, policy)
    if key in cache:
        # This reuses a completed memory state, but must account for logical pipeline cost on repeated QAs.
        m, c, op = cache[key]
        for name, amount in c.items(): meter[name] += amount
        return [x.copy() for x in m], op.copy()
    before = meter.copy()
    mems, op = [], collections.Counter()
    events = []
    checkpoint_path = None
    checkpoint_events = []
    if case.get('dataset') == 'locomo' and getattr(llm,'checkpoint_root',None) is not None and policy != 'add_only':
        audit_dir = Path(llm.checkpoint_root) / 'history_audit'
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_key = sha({'history':key[0],'batch':batch,'policy':policy})
        checkpoint_path = audit_dir / (audit_key + '.partial.json')
        if checkpoint_path.exists():
            try: checkpoint_events=json.loads(checkpoint_path.read_text(encoding='utf-8')).get('events',[])
            except (json.JSONDecodeError,OSError): checkpoint_events=[]
            valid=0
            for saved,ch in zip(checkpoint_events,chunks(case['turns'],batch)):
                if saved.get('chunk_turn_ids') != [t['id'] for t in ch]: break
                valid+=1
            checkpoint_events=checkpoint_events[:valid]
    case_chunks=chunks(case['turns'],batch)
    for num,ch in enumerate(case_chunks,1):
        saved=checkpoint_events[num-1] if num <= len(checkpoint_events) else None
        if saved is not None and policy == 'add_only':
            valid_ids={t['id'] for t in ch}
            fresh_rows=saved.get('decision',{}).get('fresh',[])
            new=[]
            for fact in fresh_rows:
                if 'source_turn_ids' in fact:
                    source_ids=fact.get('source_turn_ids',[])
                else:
                    idx=fact.get('index',0)
                    source_ids=[ch[idx]['id']] if type(idx) is int and 0 <= idx < len(ch) else []
                new.append({'text':fact['text'],'source_ids':[sid for sid in source_ids if sid in valid_ids],
                            'date':ch[-1]['date']})
            changes=apply_add_only(mems,new,ch)
            decision=saved.get('decision',{})
        else:
            new=extract(llm,ch,meter)
            changes=update(llm,policy,mems,new,ch,meter,topk)
            decision=changes.pop('trace')
            event={'chunk_turn_ids':[t['id'] for t in ch],'date':ch[-1]['date'],'decision':decision}
            if saved is not None: checkpoint_events=checkpoint_events[:num-1]
            checkpoint_events.append(event)
            if checkpoint_path:
                temp=checkpoint_path.with_suffix('.tmp')
                temp.write_text(jdump({'group':case.get('group',''),'batch':batch,'policy':policy,
                                       'events':checkpoint_events}),encoding='utf-8')
                temp.replace(checkpoint_path)
        events.append({'chunk_turn_ids':[t['id'] for t in ch],'date':ch[-1]['date'],'decision':decision})
        op.update(changes)
    if checkpoint_path:
        if policy == 'add_only' or len(events)>=len(case_chunks):
            if checkpoint_path.exists(): checkpoint_path.unlink()
        else:
            temp=checkpoint_path.with_suffix('.tmp')
            temp.write_text(jdump({'group':case.get('group',''),'batch':batch,'policy':policy,
                                   'events':events}),encoding='utf-8')
            temp.replace(checkpoint_path)
    if getattr(llm,'path',None) is not None:
        audit_dir = llm.path.parent / 'history_audit'; audit_dir.mkdir(parents=True, exist_ok=True)
        audit_key = sha({'history':key[0],'batch':batch,'policy':policy})
        (audit_dir / (audit_key + '.json')).write_text(jdump({'group':case.get('group',''),
            'batch':batch,'policy':policy,'events':events}),encoding='utf-8')
    deltas = {k: meter[k]-before[k] for k in before}
    cache[key] = ([x.copy() for x in mems], deltas, dict(op))
    return mems, dict(op)


def answer(llm, case, mems, meter, topk):
    # Query = question directly: same QA retriever and top-k for all experiment arms.
    retrieval = bm25_rank(case['question'], mems, topk)
    system = ("Answer the question using ONLY the provided memory facts. Pay attention to dates, changed preferences "
              "and current vs historical state. If insufficient evidence, answer 'unknown'. Output concise answer JSON. "
              "Do not use prior general knowledge to fill missing facts.")
    obj = llm.run('answer_qa', system, {'question': case['question'],
                    'memories': [{'text': m['text'], 'date': m['date']} for m in retrieval]},
                  ANSWER_SCHEMA, meter)
    pred = str(obj['answer']).strip()
    judge_system = ("Judge whether PREDICTION correctly answers QUESTION relative to GOLD ANSWER. "
                    "Allow semantic paraphrases, equivalent date expressions, and extra harmless details; "
                    "reject contradictions, unsupported guesses, and answers that miss essential elements. "
                    "If GOLD is unanswerable, accept an explicit abstention. Return only boolean correct.")
    verdict = llm.run('judge_qa', judge_system, {'question':case['question'],
                      'gold':case['answer'], 'prediction':pred}, JUDGE_SCHEMA, meter)
    gold_tokens, pred_tokens = tokenize(case['answer']), tokenize(pred)
    overlap = sum((collections.Counter(gold_tokens) & collections.Counter(pred_tokens)).values())
    precision = overlap/len(pred_tokens) if pred_tokens else 0.
    recall = overlap/len(gold_tokens) if gold_tokens else 0.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    evidence = set(case['evidence'])
    covered = evidence.intersection({sid for m in retrieval for sid in m.get('source_ids', [])})
    return {'prediction':pred, 'correct':bool(verdict['correct']),
            'em':int(normalize(case['answer'])==normalize(pred)), 'f1':f1,
            'evidence_origin_recall': (len(covered)/len(evidence) if evidence else None),
            'retrieved_ids': [m['id'] for m in retrieval]}


def summarize(rows):
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['stage'],r['arm'],r['dataset'])].append(r)
    for r in rows:
        groups[(r['stage'],r['arm'],'OVERALL')].append(r)
        if r['dataset']=='longmemeval' and r['type']=='knowledge-update':
            groups[(r['stage'],r['arm'],'longmemeval:knowledge-update')].append(r)
    out = []
    for (stage, arm, dataset), rr in sorted(groups.items()):
        def avg(key): return round(statistics.mean([float(x[key]) for x in rr]), 5)
        x = {'stage':stage,'arm':arm,'dataset':dataset,'n':len(rr),
             'correct_rate':avg('correct'),'em':avg('em'),'token_f1':avg('f1'),
             'mean_memory_count':avg('memory_count'),'mean_llm_calls':avg('calls'),
             'mean_input_tokens':avg('input_tokens'),'mean_output_tokens':avg('output_tokens'),
             'mean_model_latency_s':avg('latency_s'),'mean_estimated_cost_usd':avg('estimated_cost_usd'),
             'mean_added':avg('added'),
             'mean_deleted':avg('deleted'),'mean_skipped':avg('skipped'),
             'invalid_indices':sum(int(v['invalid_indices']) for v in rr)}
        origin = [r['evidence_origin_recall'] for r in rr if r['evidence_origin_recall'] is not None]
        x['evidence_origin_recall'] = round(statistics.mean(origin),5) if origin else None
        out.append(x)
    return out


def write_csv(path, records):
    if not records: return
    with open(path,'w', newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


def evaluate(llm, selected_cases, stage, arms, state_cache, answer_topk, update_topk, out_file):
    rows = []
    completed_arms = collections.defaultdict(set)
    if Path(out_file).exists() and Path(out_file).stat().st_size:
        with Path(out_file).open(encoding='utf-8') as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if row.get('stage') == stage and row.get('case_id') in {c['id'] for c in selected_cases}:
                    completed_arms[row['case_id']].add(row['arm'])
                    rows.append(row)
        print(f'[{stage}] resume: {sum(len(v) for v in completed_arms.values())} arm rows found', flush=True)
    mode = 'a' if Path(out_file).exists() and Path(out_file).stat().st_size else 'w'
    with open(out_file, mode, encoding='utf-8') as f:
        for num, case in enumerate(selected_cases, 1):
            for arm, batch, policy in arms:
                if arm in completed_arms[case['id']]:
                    continue
                meter = new_meter()
                mems, stats = make_history(llm, case, batch, policy, meter, state_cache, update_topk)
                res = answer(llm, case, mems, meter, answer_topk)
                cost = ((meter['input_tokens']-meter['cached_input_tokens'])*llm.price_in
                        + meter['cached_input_tokens']*getattr(llm, 'cached_price',llm.price_in)
                        + meter['output_tokens']*llm.price_out)/1_000_000
                row = {'stage':stage, 'arm':arm, 'dataset':case['dataset'], 'type':case['type'],
                       'case_id':case['id'], 'history_group':case['group'], 'answer_gold':case['answer'], **res,
                       'memory_count':len(mems), 'estimated_cost_usd':round(cost,8), **meter,
                       **{key:stats.get(key, 0) for key in ['added','deleted','skipped','invalid_indices']}}
                f.write(jdump(row) + '\n'); f.flush()
                rows.append(row)
                completed_arms[case['id']].add(arm)
            print(f'[{stage}] {num}/{len(selected_cases)}  {case["id"]}', flush=True)
    return rows


def bootstrap_ci_delta(rows, a, b, seed=42, n_boot=2000):
    """Paired CLUSTER bootstrap: same LoCoMo dialogue QAs are not independent."""
    byarm = collections.defaultdict(dict)
    for r in rows: byarm[r['arm']][r['case_id']] = r
    common = sorted(set(byarm[a]).intersection(byarm[b]))
    if not common: return None
    clusters = collections.defaultdict(list)
    for i in common:
        assert byarm[a][i]['history_group'] == byarm[b][i]['history_group']
        clusters[byarm[a][i]['history_group']].append(int(byarm[a][i]['correct']) - int(byarm[b][i]['correct']))
    group_ids = sorted(clusters)
    diffs = [x for cid in group_ids for x in clusters[cid]]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choices(group_ids, k=len(group_ids))
        sampled = [v for cid in draw for v in clusters[cid]]
        vals.append(100*sum(sampled)/len(sampled))
    vals.sort()
    return {'a':a,'b':b,'paired_n':len(common),'independent_clusters':len(group_ids),
            'delta_pp':round(100*sum(diffs)/len(diffs),2),
            'cluster_bootstrap_ci95_pp':[round(vals[int(n_boot*.025)],2),round(vals[int(n_boot*.975)],2)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='gpt-6-luna')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output-dir',default='results')
    p.add_argument('--exp2-n', type=int, default=80,
                   help='Number of held-out cases to evaluate in exp2 (up to all available cases)')
    p.add_argument('--exp2-outdir',default=None,
                   help='Directory for an independent exploratory exp2 run')
    p.add_argument('--exp2-order', choices=('manifest','interleave'), default='manifest',
                   help='Evaluation order; interleave datasets to avoid long runs starting with only LoCoMo')
    p.add_argument('--exp2-order-start', choices=('locomo','longmemeval'), default='locomo',
                   help='First dataset when --exp2-order interleave is used')
    p.add_argument('--exp2-only',action='store_true', help='Skip exp1 and reuse its existing results')
    p.add_argument('--fresh-exp2-output',default=None,
                   help='Write this exp2 run to a new JSONL path, preserving previous partial files')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topk',type=int,default=8)
    p.add_argument('--update-topk',type=int,default=12)
    p.add_argument('--input-price',type=float,default=0.10,help='USD / 1M input tokens, standard short context')
    p.add_argument('--output-price',type=float,default=0.50,help='USD / 1M output tokens, standard short context')
    p.add_argument('--cached-input-price',type=float,default=0.01,help='USD / 1M cached input tokens')
    p.add_argument('--offline',action='store_true',help='Only cached model calls; never access API')
    args = p.parse_args()
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True,exist_ok=True)
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    exp2_out = Path(args.exp2_outdir) if args.exp2_outdir else out
    if not exp2_out.is_absolute(): exp2_out = (Path.cwd()/exp2_out).resolve()
    exp2_out.mkdir(parents=True,exist_ok=True)
    for name, url in DATA_URLS.items():
        dst = data_dir / (name+'.json')
        if not dst.exists():
            if args.offline: raise FileNotFoundError(dst)
            print('Downloading', url, flush=True)
            urlretrieve(url, dst)
    lo = build_locomo(json.loads((data_dir/'locomo.json').read_text(encoding='utf-8')))
    lm = build_longmemeval(json.loads((data_dir/'longmemeval.json').read_text(encoding='utf-8')))
    dev, test = sample_cases(lo,lm,args.seed)
    if len(dev)!=20 or len(test)<80: raise AssertionError('Expected 20 dev cases and at least 80 held-out cases')
    if not 1 <= args.exp2_n <= len(test):
        raise ValueError(f'--exp2-n must be between 1 and {len(test)}')
    if args.exp2_n > 80 and args.exp2_outdir is None:
        raise ValueError('For exp2-n > 80, set --exp2-outdir to a separate exploratory output directory')
    test = test[:args.exp2_n]
    if args.exp2_order == 'interleave':
        queues = collections.defaultdict(collections.deque)
        for case in test: queues[case['dataset']].append(case)
        names = ['locomo', 'longmemeval']
        if args.exp2_order_start == 'longmemeval': names.reverse()
        names = [name for name in names if queues[name]]
        interleaved = []
        while any(queues[name] for name in names):
            for name in names:
                if queues[name]: interleaved.append(queues[name].popleft())
        test = interleaved
    exp2_path = Path(args.fresh_exp2_output).resolve() if getattr(args,'fresh_exp2_output',None) else exp2_out/'exp2_test.jsonl'
    manifest_path = exp2_out/'manifest.json' if exp2_out == out else exp2_out/'exp2_test.manifest.json'
    if getattr(args,'fresh_exp2_output',None): manifest_path=Path(args.fresh_exp2_output).with_suffix('.manifest.json')
    with open(manifest_path,'w',encoding='utf-8') as f:
        json.dump({'seed':args.seed,'model':args.model,'exp2_order':args.exp2_order,
                   'exp2_order_start':args.exp2_order_start,'dev':[c['id'] for c in dev],
                   'test':[c['id'] for c in test], 'data_sources':DATA_URLS},f,ensure_ascii=False,indent=2)
    cache_root = exp2_out/'api_cache' if args.exp2_outdir else out/'api_cache'
    llm = LLM(args.model, cache_root,args.input_price,args.output_price,args.offline)
    if args.exp2_outdir and not args.offline:
        llm.checkpoint_root = exp2_out
    llm.cached_price = args.cached_input_price
    state_cache = {}
    # Exp 1: exact same storage policy (no semantic dedup/delete), QA recall and model cost.
    if args.exp2_only and not (out/'exp1_dev.jsonl').exists():
        raise FileNotFoundError(f'--exp2-only requested, but {(out/"exp1_dev.jsonl")} is missing')
    if (args.exp2_only or args.exp2_outdir or getattr(args,'fresh_exp2_output',None)) and (out/'exp1_dev.jsonl').exists():
        one_rows = [json.loads(line) for line in (out/'exp1_dev.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        one_rows = [row for row in one_rows if row.get('stage') == 'exp1']
    else:
        one_rows = evaluate(llm,dev,'exp1', [('turn_1',1,'add_only'), ('turn_5',5,'add_only')],
                            state_cache,args.topk,args.update_topk,out/'exp1_dev.jsonl')
    s = summarize(one_rows)
    dev_agg = {r['arm']:r for r in s if r['dataset']=='OVERALL'}
    # Pre-registered selection: QA accuracy, then token-F1, then fewer calls.
    best = max(['turn_1','turn_5'], key=lambda a:(dev_agg[a]['correct_rate'],
            dev_agg[a]['token_f1'],-dev_agg[a]['mean_llm_calls']))
    win_batch = 1 if best=='turn_1' else 5
    print('Exp1 winner on dev (not test):',best,flush=True)
    arms = [('joint',win_batch,'joint'),('sequential',win_batch,'sequential'),
            ('delete_only',win_batch,'delete_only'),('add_only',win_batch,'add_only')]
    two_rows = evaluate(llm,test,'exp2',arms,state_cache,args.topk,args.update_topk,exp2_path)
    agg = summarize(one_rows+two_rows)
    if exp2_out == out: write_csv(out/'summary.csv', agg)
    report = {'winner_exp1_on_dev':best, 'batch_used_exp2':win_batch,
              'total_unique_questions':len(dev)+len(test), 'exp1_dev_n':len(dev), 'exp2_heldout_n':len(test),
              'physical_api_calls':llm.actual_api_calls,
              'pairs': [bootstrap_ci_delta(two_rows, a, 'add_only',args.seed)
                        for a in ['joint','sequential','delete_only']],
              'notes':['Per-case output and model calls are cached to disk.',
                       'Costs are logical per arm, not incremental physical cache-hit charges.',
                       'Model-judged QA correctness is not an independent human assessment.',
                       'LoCoMo has 10 base dialogues: QAs within a dialogue are correlated.',
                       'LongMemEval oracle uses only evidence sessions, not full noisy histories.']}
    report_path=exp2_out/'report.json'
    if getattr(args,'fresh_exp2_output',None): report_path=Path(args.fresh_exp2_output).with_suffix('.report.json')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('Summary:',out/'summary.csv' if exp2_out == out else 'not updated for exploratory output')

if __name__=='__main__': main()
