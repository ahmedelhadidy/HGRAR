import pandas as pd
from model.grar.gRule import GRule
from model.grar.term import Term
from model.grar.item import Item
from model.grar.operator import OperatorType
from model.grar.arithmetic_operator import ArithmeticOperator
from model.grar.operator import Operator
import model.grar.gRule as gRule



def start(dataset, operators, min_s, min_c, min_membership, rule_max_length = None):
    candidate_rules = __generate_initial_candidates(dataset, operators)
    interesting_r = __scan_get_interesting_rules(candidate_rules,dataset,min_s, min_c, min_membership)
    final_interesting_r = interesting_r
    k = 2
    m = len(dataset.columns) if not rule_max_length else rule_max_length
    while len(interesting_r) > 1 and k < m:
        candidate_rules = __generate_new_candidates(list([r[0] for r in interesting_r]), interesting_r[0][0].length+1)
        interesting_r = __scan_get_interesting_rules(candidate_rules, dataset, min_s, min_c, min_membership)
        if len(interesting_r) > 0:
            final_interesting_r = interesting_r
        k += 1
    return final_interesting_r


def __generate_initial_candidates(dataset, operators):
    heads = list(dataset.columns.values)
    cand_rules = list([__create_binary_rule(item1, item2, op)
                       for item1 in heads
                       for item2 in heads[heads.index(item1)+1:]
                       for op in operators if op.operator_type != OperatorType.NE

    ])
    return cand_rules


def __create_binary_rule(t1,t2,op):
    rule = GRule(2)
    item_1 = Item(t1,1)
    item_2 = Item(t2,2)
    rule.add_term(Term(item_1,op))
    rule.add_term(Term(item_2, Operator(OperatorType.NE)))
    return rule


def __scan_get_interesting_rules(candidate_rules,dataset, min_s, min_c, min_membership):
    interesting_rules = []
    for rule in candidate_rules:
        s,c, m = __calculate_support_confidence_membership(rule, dataset, min_membership)
        if s >= min_s and c >= min_c:
            interesting_rules.append((rule,m))
    return interesting_rules


def __calculate_support_confidence_membership(rule:GRule, dataset, min_membership):
    s_counter =0
    c_counter=0
    m_c=0
    n = dataset.shape[0]
    for index,row in dataset.iterrows():
        if __all_rule_items_present(row, rule.get_all_items()):
            s_counter+=1
            membership_degree = rule.calculate_membership_degree(row)
            if membership_degree>= min_membership:
                c_counter+=1
                m_c += membership_degree
    return s_counter/n, c_counter/n, m_c/n


def __all_rule_items_present(data_row, items):
    for item in items:
        if not data_row[item.category]:
            return False
    return True


def __generate_new_candidates(grules, new_length):
    new_candidate = []
    for index, rule1 in enumerate(grules):
        remainning_rules_list = grules[index + 1:]
        if len(remainning_rules_list) == 0:
            break
        for rule2 in remainning_rules_list:
            join_rule, terms = gRule.get_join_rule(rule1, rule2)
            if None not in terms:
                new_terms = gRule.join(join_rule, *terms)
                new_grule = GRule(new_length)
                new_grule.add_items(new_terms)
                new_candidate.append(new_grule)
    return new_candidate


def read_dataset(file):
    return pd.read_csv(file, index_col=False)


def test_arithmetic_grar():
    dataset = read_dataset('test_data/test_dataset')
    use_fuzzt = True
    min_s = 1
    min_c = 0.9
    min_membership = 0.8
    operators = [ArithmeticOperator(OperatorType.LTE, use_fuzzt), ArithmeticOperator(OperatorType.GTE, use_fuzzt),
                 ArithmeticOperator(OperatorType.EQ, use_fuzzt)]
    interesting_rules = start(dataset, operators, min_s, min_c, min_membership, 2)
    for r, m in interesting_rules:
        print("========")
        r_stt = '('
        for t in r.g_rule_set:
            r_stt += str(t) + " "
        r_stt += ')'
        print('grule %s  grule membership  %s ' % (r_stt, m))


def _print_interesting_rule(grars, member_ship_degree_threshold = 0.5):
    for r, m in grars:
        if m < member_ship_degree_threshold:
            continue
        print("========")
        r_stt = '('
        for t in r.g_rule_set:
            r_stt += str(t) + " "
        r_stt += ')'
        print('grule %s  grule membership  %s ' % (r_stt, m))

