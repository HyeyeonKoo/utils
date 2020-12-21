#-*-coding:utf-8-*-

from abc import abstractmethod
from copy import deepcopy
from konlpy.tag import Kkma
import math


class IrEval:
    def __init__(self, query, result, correct, match="exact", threshold=0.55):
        if self.check_type_query(query):
            self.query = query
        if self.check_type_result(result):
            self.result = result
        if self.check_type_result(correct):
            self.correct = correct

        self.detail = {}
        self.score = None

        self.match = match
        self.threshold = threshold


    @abstractmethod
    def eval(self):
        pass


    def check_same(self, a, b):
        if self.match == "exact":
            if a == b:
                return True
            else:
                return False
        
        elif self.match == "jaccard":
            if self.cal_jaccard(a, b):
                return True
            else:
                return False

        else:
            raise RuntimeError("Match mode must be 'exact' or 'jaccard'.")


    def cal_jaccard(self, a, b):
        tokenizer = Kkma()
        a_ = set(tokenizer.morphs(a))
        b_ = set(tokenizer.morphs(b))
        jaccard = len(a_ & b_ ) / len(a_ | b_)
        return jaccard >= self.threshold

        
    def check_type_query(self, query):
        if type(query) is not list:
            raise RuntimeError("The query must be list.")
        return True
            
            
    def check_type_result(self, result):
        if type(result) is not list or type(result[0]) is not list:
            raise RuntimeError("The object and element of result must be list.")
        return True


class MRR(IrEval):
    def __init__(self, query, result, correct, match="exact", threshold=0.55):
        super(MRR, self).__init__(query, result, correct, match, threshold)
        
        
    def eval(self):
        reciprocal = []
        
        for i in range(len(self.query)):
            self.detail[self.query[i]] = {"result":{}, "score":None}

            score = self.reciprocal_rank(self.query[i], self.result[i], self.correct[i])
            reciprocal.append(score)
            self.detail[self.query[i]]["score"] = score
            
        self.score = sum(reciprocal) / len(reciprocal)
            
            
    def reciprocal_rank(self, query_, result_, correct_):
        index = -1
        n = len(result_)
        
        for i in range(len(self.correct)):
            break_check = False
            
            for j in range(n):
                if self.check_same(correct_[i], result_[j]):
                    self.detail[query_]["result"][result_[j]] = j
                    index = i
                    break_check = True
                    break
                    
            if break_check:
                break
        
        return (n - index) / n


class MAP(IrEval):
    
    def __init__(self, query, result, correct,  match="exact", threshold=0.55):
        super(MAP, self).__init__(query, result, correct, match, threshold)
        
    
    def eval(self):
        avg_precision = []

        for i in range(len(self.query)):
            self.detail[self.query[i]] = {"result":{}, "score":None}

            score = self.avg_precision(self.query[i], self.result[i], self.correct[i])
            avg_precision.append(score)
            self.detail[self.query[i]]["score"] = score

        self.score = sum(avg_precision) / len(avg_precision)
    
    
    def avg_precision(self, query_, result_, correct_):
        precision = []

        relevant = self.get_relevant(result_, correct_)
        check_sum = 0

        for i in range(len(relevant)):
            if relevant[i]:
                check_sum += 1
                precision_score = check_sum / (i + 1)

                precision.append(precision_score)
                self.detail[query_]["result"][result_[i]] = precision_score

        return sum(precision) / len(precision)


    def get_relevant(self, result_, correct_):
        relevant = [False] * len(result_)

        for i in range(len(result_)):
            check = False

            for j in range(len(correct_)):
                if self.check_same(result_[i], correct_[j]):
                    check = True
                    break

            if check:
                relevant[i] = True

        return relevant
                    
        
class NDCG(IrEval):
    
    def __init__(self, query, result, correct, rank_score, match="exact", threshold=0.55):
        super(NDCG, self).__init__(query, result, correct, match, threshold)

        if self.check_rank_score(rank_score):
            self.rank_score = rank_score
            self.correct_dcg = self.get_correct_dcg()


    def check_rank_score(self, rank_score):
        for element in rank_score:
            if type(element) is str:
                raise RuntimeError("The rank_score must be number.")

        if len(self.result) != len(self.correct) != len(rank_score):
            raise RuntimeError("In NDCG case, result, correct, rank_score must have same length.")

        return True


    def eval(self):
        ndcg = []

        for i in range(len(self.query)):
            self.detail[self.query[i]] = {"result":{}, "score":None}

            result_dcg = self.get_result_dcg(self.query[i], self.result[i], self.correct[i])

            score = result_dcg / self.correct_dcg
            ndcg.append(score)
            self.detail[self.query[i]]["score"] = score

        self.score = sum(ndcg) / len(ndcg)


    def get_correct_dcg(self):
        dcg = []

        for i in range(len(self.rank_score)):
            dcg.append(self.rank_score[i] / math.log((i + 1) + 1, 2))

        return sum(dcg)


    def get_result_dcg(self, query_, result_, correct_):
        dcg = []

        for i in range(len(result_)):
            for j in range(len(correct_)):
                if self.check_same(result_[i], correct_[j]):
                    score = self.rank_score[j] / math.log((i + 1) + 1, 2)
                    dcg.append(score)
                    self.detail[query_]["result"][result_[i]] = score

        return sum(dcg)
