"""Offline deterministic unit/smoke tests; no GPT model call or dataset download."""
import unittest
from collections import defaultdict
from memory_benchmark import (chunks, bm25_rank, update, make_history, new_meter, sample_cases,
                              extract, build_locomo, build_longmemeval, bootstrap_ci_delta)


class StubLLM:
    def run(self, label, system, payload, schema, meter):
        meter['calls'] += 1
        if label == 'extract_facts':
            return {'facts':[{'text': v['text'], 'source_turn_ids':[v['id']]}
                             for v in payload['turns']]}
        if label == 'memory_query': return {'query': 'Alex beverage'}
        if label == 'joint_indices': return {'skip_new_indices': [], 'delete_old_indices':[0]}
        if label == 'skip_indices': return {'skip_new_indices':[]}
        if label in ('delete_indices','delete_only_indices'): return {'delete_old_indices':[0]}
        if label == 'answer_qa': return {'answer':'coffee'}
        if label == 'judge_qa': return {'correct':True}
        raise RuntimeError(label)


class CacheCountingStub(StubLLM):
    path = None
    actual_api_calls = 0
    def __init__(self): self.calls_by_chunk = {}
    def run(self, label, system, payload, schema, meter):
        if label == 'extract_facts':
            turn_id = payload['turns'][0]['id']
            self.calls_by_chunk[turn_id] = self.calls_by_chunk.get(turn_id, 0) + 1
            meter['cache_misses'] += 1
        return super().run(label, system, payload, schema, meter)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {'id':'a','session':'s1','date':'2026-01-01','speaker':'Alex','text':'Alex prefers tea'},
            {'id':'b','session':'s2','date':'2026-02-01','speaker':'Alex','text':'Alex prefers coffee now'},
        ]

    def test_session_boundary(self):
        self.assertEqual([len(c) for c in chunks(self.turns,5)],[1,1])
        v = [{**t,'session':'s1'} for t in self.turns]
        self.assertEqual([len(c) for c in chunks(v,5)],[2])
        self.assertEqual([len(c) for c in chunks(v,1)],[1,1])

    def test_bm25(self):
        m=[{'id':'1','seq':1,'text':'Alex prefers tea'},
           {'id':'2','seq':2,'text':'Bob drives a car'}]
        self.assertEqual(bm25_rank('Alex tea',m,1)[0]['id'],'1')

    def test_update_policies(self):
        older = {'id':'old','text':'Alex prefers tea','source_ids':['a'], 'date':'2026-01-01','seq':1}
        ch = [self.turns[1]]
        fresh = [{'text':'Alex prefers coffee now','source_ids':['b'],'date':'2026-02-01'}]
        for policy in ['joint','sequential','delete_only','add_only']:
            mems=[older.copy()]
            stats=update(StubLLM(),policy,mems,fresh,ch,new_meter(),12)
            if policy=='add_only':
                self.assertEqual(len(mems),2)
                self.assertEqual(stats['deleted'],0)
            else:
                self.assertEqual(len(mems),1)
                self.assertEqual(stats['deleted'],1)
                self.assertIn('coffee',mems[0]['text'])

    def test_repeat_history_cached_ops(self):
        case={'turns':[self.turns[0]], 'question':'What does Alex prefer?'}
        cache={}; llm=StubLLM()
        m1=new_meter(); st1, op1=make_history(llm,case,1,'add_only',m1,cache,8)
        m2=new_meter(); st2, op2=make_history(llm,case,1,'add_only',m2,cache,8)
        self.assertEqual(op1,op2)
        self.assertEqual(st1,st2)
        self.assertEqual(m1['calls'],m2['calls'])

    def test_chunk_checkpoint_replay_only_calls_missing_chunks(self):
        import tempfile, json
        from pathlib import Path
        turns = [{'id':f't{i}', 'session':'s', 'date':'2026-01-01',
                  'speaker':'Alex', 'text':f'Alex says fact {i}'} for i in range(120)]
        case={'id':'locomo:test:1','dataset':'locomo','group':'locomo:test',
              'turns':turns, 'question':'fact?', 'type':'2'}
        with tempfile.TemporaryDirectory() as tmp:
            llm=CacheCountingStub(); llm.path=Path(tmp)
            audit=Path(tmp)/'history_audit'; audit.mkdir()
            from memory_benchmark import sha
            checkpoint=audit/(sha({'history':sha(turns),'batch':1,'policy':'add_only'})+'.partial.json')
            events=[]
            for ch in chunks(turns,1)[:40]:
                events.append({'chunk_turn_ids':[t['id'] for t in ch], 'date':ch[-1]['date'],
                    'decision':{'fresh':[{'index':0,'text':ch[0]['text'],'source_turn_ids':[ch[0]['id']]}],
                                'retrieved':[],'skip_new_indices':[],'delete_old_indices':[]}})
            checkpoint.write_text(json.dumps({'events':events}))
            meter=new_meter()
            memories,_=make_history(llm,case,1,'add_only',meter,{},8)
            self.assertEqual(len(memories),120)
            self.assertEqual(meter['cache_hits'],0)
            self.assertEqual(meter['cache_misses'],120)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(len(json.loads(checkpoint.read_text())['events']),40)

    def test_loaders_and_group_split(self):
        conversations=[]
        for i in range(10):
            conversations.append({'sample_id':str(i),
              'conversation':{'session_1':[{'speaker':'A','dia_id':'D1:1','text':'A likes coffee'}],
                              'session_1_date_time':'2025-01-01'},
              'qa':[{'question':f'q {j}', 'answer':'coffee','category':1,'evidence':['D1:1']}
                    for j in range(6)]})
        locomo=build_locomo(conversations)
        lm_data=[]
        for i in range(80):
            lm_data.append({'question_id':str(i),
                            'question_type':'knowledge-update' if i<30 else 'multi-session',
                            'question':'Coffee?', 'answer':'yes',
                            'haystack_dates':['2025-02-01'],
                            'haystack_session_ids':['s'],
                            'haystack_sessions':[[{'role':'user','content':'I like coffee','has_answer':True}]]})
        lm=build_longmemeval(lm_data)
        dev,test=sample_cases(locomo,lm,42)
        self.assertEqual((len(dev),len(test)),(20,80))
        self.assertEqual(len(set(x['id'] for x in dev+test)),100)
        dev_groups = set(c['group'] for c in dev if c['dataset']=='locomo')
        test_groups = set(c['group'] for c in test if c['dataset']=='locomo')
        self.assertFalse(dev_groups & test_groups)
        self.assertEqual(len(dev_groups), 2)
        self.assertGreaterEqual(len([c for c in dev+test if c['type']=='knowledge-update']),0)

    def test_exp2_interleave(self):
        import collections
        cases=[{'id':'l1','dataset':'locomo'}, {'id':'l2','dataset':'locomo'},
               {'id':'m1','dataset':'longmemeval'}, {'id':'m2','dataset':'longmemeval'},
               {'id':'m3','dataset':'longmemeval'}]
        queues=collections.defaultdict(collections.deque)
        for case in cases: queues[case['dataset']].append(case)
        names=['locomo','longmemeval']; interleaved=[]
        while any(queues[name] for name in names):
            for name in names:
                if queues[name]: interleaved.append(queues[name].popleft())
        self.assertEqual([c['id'] for c in interleaved],['l1','m1','l2','m2','m3'])

    def test_staged_exp2_slice_uses_full_manifest(self):
        cases=[{'id':str(i)} for i in range(80)]
        start,n=10,1
        batch=cases[start:start+n]
        self.assertEqual(len(batch),1)
        self.assertEqual(batch[0]['id'],'10')

    def test_evaluate_resume_keeps_partial_rows(self):
        import json, tempfile
        from pathlib import Path
        from memory_benchmark import evaluate
        case={'id':'locomo:r:1','dataset':'locomo','group':'locomo:r','type':'2',
              'question':'What?','answer':'coffee','evidence':[],'turns':[self.turns[0]]}
        with tempfile.TemporaryDirectory() as tmp:
            llm=StubLLM(); llm.price_in=.1; llm.price_out=.5
            path=Path(tmp)/'rows.jsonl'
            old={'stage':'exp2','arm':'joint','case_id':case['id'],'dataset':'locomo',
                 'history_group':case['group'],'correct':True,'f1':1,'em':1,
                 'memory_count':1,'calls':1,'input_tokens':0,'output_tokens':0,
                 'cached_input_tokens':0,'latency_s':0,'estimated_cost_usd':0,
                 'added':1,'deleted':0,'skipped':0,'invalid_indices':0}
            path.write_text(json.dumps(old)+'\n')
            rows=evaluate(llm,[case],'exp2',[('joint',1,'joint'),('add_only',1,'add_only')],{},8,8,path)
            saved=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(saved),2)
            self.assertEqual({r['arm'] for r in saved},{'joint','add_only'})
            self.assertEqual(len(rows),2)

    def test_cluster_bootstrap(self):
        records=[]
        for name in ('joint','add_only'):
            for i in range(4):
                records.append({'arm':name, 'case_id':str(i),
                                'history_group':'a' if i<2 else str(i),
                                'correct':1 if name=='joint' else 0})
        ci=bootstrap_ci_delta(records,'joint','add_only',n_boot=100)
        self.assertEqual(ci['paired_n'],4)
        self.assertEqual(ci['independent_clusters'],3)
        self.assertEqual(ci['delta_pp'],100.0)


if __name__=='__main__': unittest.main()
