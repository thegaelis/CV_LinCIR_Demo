import unittest
from core import load_cases, case_ids, candidates, required_ids

class GeneCISTest(unittest.TestCase):
    def test_official_protocol_and_nested_selection(self):
        for task in ('focus_object','change_object'):
            a=load_cases(task,20); b=load_cases(task,100)
            self.assertEqual(a,b[:20]); self.assertEqual(len(a),20)
            for _,case in a:
                ref,target,gallery=case_ids(case)
                self.assertEqual(len(gallery),14)
                self.assertEqual(len(candidates(case)),15)
                self.assertIn(target,candidates(case))
                self.assertIn(ref,required_ids(a))
                self.assertEqual(candidates(case),candidates(case))
    def test_invalid(self):
        with self.assertRaises(ValueError): load_cases('attribute',1)
        with self.assertRaises(ValueError): load_cases('focus_object',0)
if __name__=='__main__': unittest.main()
