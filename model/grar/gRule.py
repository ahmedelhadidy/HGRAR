from model.grar.term import  Term
from model.grar.exceptions import ExceedingGRuleLength
from model.grar.operator import Operator, OperatorType
import copy

class GRule:

    def __init__(self, length):
        self.g_rule_set = []
        self.length = length

    def add_term(self, term: Term):
        if len(self.g_rule_set) == self.length:
            raise ExceedingGRuleLength('Exceeding defined GRule length %s'.format(self.length))
        self.g_rule_set.append(term)

    def add_items(self, items):
        if len(self.g_rule_set) + len(items) > self.length:
            raise ExceedingGRuleLength('Exceeding defined GRule length %s'.format(self.length))
        self.g_rule_set.extend(items)

    def get_length(self):
        return self.length

    def calculate_membership_degree(self, data_row):
        min_member_ship_degree = 1
        terms_tubles_list = list([ (self.g_rule_set[i], self.g_rule_set[i+1])
                                 for i in range(len(self.g_rule_set)-1)
                                 if i < len(self.g_rule_set)
                                  ])
        for term1, term2 in terms_tubles_list:
            member_ship_degree = term1.apply(term2, data_row)
            if member_ship_degree < min_member_ship_degree:
                min_member_ship_degree = member_ship_degree
        return min_member_ship_degree

    def get_all_items(self):
        return list([term.item for term in self.g_rule_set])

    def join_r1_detected(self, grule2):
        grule1_terms = copy.deepcopy(self.g_rule_set)
        grule2_terms = copy.deepcopy(grule2.g_rule_set)
        grule1_prefex = []
        grule2_suffex = [Term(None, Operator(OperatorType.NE))]

        while len(grule1_terms) > 1 :
            matched = True
            grule1_prefex.append(grule1_terms[0])
            grule1_terms = grule1_terms[1:]

            grule2_suffex.insert(1, grule2_terms[-1])
            grule2_suffex[0].operator = grule2_terms[-2].operator
            grule2_terms = grule2_terms[:-1]
            grule2_terms[-1].operator = Operator(OperatorType.NE)
            for index, term in enumerate(grule1_terms):
                if term != grule2_terms[index]:
                    matched = False
                    break
            if matched:
                return grule1_prefex, grule1_terms, grule2_suffex
        return None, None, None

    def join_r2_detected(self, grule2):
        grule1_terms = copy.deepcopy(self.g_rule_set)
        grule2_terms = copy.deepcopy(grule2.g_rule_set)
        grule1_suffex = [Term(None, Operator(OperatorType.NE))]
        grule2_prefex = []
        while len(grule1_terms) > 1:
            matched = True
            grule1_suffex.insert(1,grule1_terms[-1])
            grule1_suffex[0].operator = grule1_terms[-2].operator
            grule1_terms = grule1_terms[:-1]
            grule1_terms[-1].operator= Operator(OperatorType.NE)

            grule2_prefex.append(grule2_terms[0])
            grule2_terms= grule2_terms[1:]

            for indx, term in enumerate(grule1_terms):
                if term != grule2_terms[indx]:
                    matched = False
                    break
            if matched:
                return grule1_suffex, grule1_terms, grule2_prefex
        return None, None, None

    def join_r3_detected(self, grule2):
        grule1_terms = copy.deepcopy(self.g_rule_set)
        grule2_terms = copy.deepcopy(grule2.g_rule_set)
        grule1_prefex=[]
        grule2_prefex=[]
        while len(grule1_terms) > 1:
            matched = True
            grule1_prefex.append(grule1_terms[0])
            grule2_prefex.append(grule2_terms[0])
            grule1_terms = grule1_terms[1:]
            grule2_terms = grule2_terms[1:]
            grule2_terms_reversed = reserv_grule_terms(grule2_terms)
            for index, term in enumerate(grule1_terms):
                if term != grule2_terms_reversed[index]:
                    matched = False
                    break
            if matched:
                return grule1_prefex, grule1_terms, grule2_prefex
        return None, None, None

    def join_r4_detected(self, grule2):
        grule1_terms = copy.deepcopy(self.g_rule_set)
        grule2_terms = copy.deepcopy(grule2.g_rule_set)
        grule1_sufex = [Term(None, Operator(OperatorType.NE))]
        grule2_sufex = [Term(None, Operator(OperatorType.NE))]
        while len(grule1_terms) > 1:
            matched = True
            grule1_sufex.insert(1, grule1_terms[-1])
            grule1_sufex[0].operator = grule1_terms[-2].operator
            grule2_sufex.insert(1, grule2_terms[-1])
            grule2_sufex[0].operator = grule2_terms[-2].operator
            grule1_terms = grule1_terms[:-1]
            grule2_terms = grule2_terms[:-1]
            grule1_terms[-1].operator = Operator(OperatorType.NE)
            grule2_terms[-1].operator = Operator(OperatorType.NE)
            rule2_reversed_terms = reserv_grule_terms(grule2_terms)
            for index, term in enumerate(grule1_terms):
                if term != rule2_reversed_terms[index]:
                    matched = False
                    break
            if matched:
                return  grule1_sufex, grule1_terms, grule2_sufex
        return None, None, None

    def __str__(self):
        instance_str =''
        for i in self.g_rule_set:
            instance_str += str(i)
        return instance_str

def is_term_in(term, terms, exclude_operator = False):
    for t in terms:
        if exclude_operator and t.item == term.item:
            return True
        elif t == term:
            return True
    return False


def is_terms_match(term_1:Term, term_2:Term, exclude_operator = False):
    return term_1.item == term_2.item if exclude_operator else term_1 == term_2


def reserv_grule_terms(terms):
    if len(terms) == 1:
        return terms
    reversed_terms = list(reversed(terms))
    return list([Term(x.item, reversed_terms[index+1].operator.revers() if index < len(reversed_terms) -1 else Operator(OperatorType.NE))
                 for index, x in enumerate(reversed_terms)])


def get_join_rule(rule1, rule2):
    terms = rule1.join_r1_detected(rule2)
    join_rule='r1'
    if None in terms:
        terms = rule1.join_r2_detected(rule2)
        join_rule='r2'
        if None in terms:
            terms=rule1.join_r3_detected(rule2)
            join_rule='r3'
            if None in terms:
                terms=rule1.join_r4_detected(rule2)
                join_rule='r4'
    return join_rule, terms


def join (join_rule, rule1_new_terms, common_terms, rule2_new_terms):
    joined_terms = []
    if join_rule == 'r1':
        joined_terms.extend(rule1_new_terms)
        joined_terms.extend(common_terms)
        joined_terms[-1].operator = rule2_new_terms[0].operator
        joined_terms.extend(rule2_new_terms[1:])
    if join_rule == 'r2':
        joined_terms.extend(rule2_new_terms)
        joined_terms.extend(common_terms)
        joined_terms[-1].operator = rule1_new_terms[0].operator
        joined_terms.extend(rule1_new_terms[1:])
    if join_rule == 'r3':
        joined_terms.extend(rule1_new_terms)
        joined_terms.extend(common_terms)
        joined_terms[-1].operator = rule2_new_terms[-1].operator.revers()
        joined_terms.extend(rule2_new_terms)
        joined_terms[-1].operator = Operator(OperatorType.NE)
    if join_rule == 'r4':
        joined_terms.extend(rule2_new_terms[1:])
        joined_terms[-1].operator = rule2_new_terms[0].operator.revers()
        joined_terms.extend(common_terms)
        joined_terms[-1].operator = rule1_new_terms[0].operator
        joined_terms.extend(rule1_new_terms[1:])
    return joined_terms


