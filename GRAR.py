import pandas as pd
from model.grar.gRule import GRule
from model.grar.term import Term
from model.grar.item import Item
from model.grar.operator import OperatorType
from model.grar.arithmetic_operator import ArithmeticOperator
from model.grar.operator import Operator
import model.grar.gRule as gRule
from itertools import combinations
import logging

LOGGER = logging.getLogger(__name__)

def start(dataset, operators, min_s, min_c, min_membership, rule_max_length = None , max_rule_count= None):

    LOGGER.info('start grar mining ')
    candidate_rules = __generate_initial_candidates(dataset, operators)
    candidate_rules = __get_max_candidate_rules_count(candidate_rules, max_rule_count)
    LOGGER.info('initial candidates count is %d '%len(candidate_rules))
    interesting_r = __scan_get_interesting_rules(candidate_rules,dataset,min_s, min_c, min_membership)
    LOGGER.info('interesting rules of length %d count is %d ' % (2, len(interesting_r)))

    final_interesting_r = interesting_r
    k = 2
    m = len(dataset.columns) if not rule_max_length else rule_max_length
    while len(interesting_r) > 1 and k < m:
        candidate_rules = __generate_new_candidates(list([r[0] for r in interesting_r]), interesting_r[0][0].length+1)
        candidate_rules = __get_max_candidate_rules_count(candidate_rules, max_rule_count)
        LOGGER.info('candidates of length %d count is %d ' % (k+1,len(candidate_rules)))
        interesting_r = __scan_get_interesting_rules(candidate_rules, dataset, min_s, min_c, min_membership)
        LOGGER.info('interesting rules of length %d count is %d ' % (k+1, len(interesting_r)))
        if len(interesting_r) > 0:
            final_interesting_r = interesting_r
        k += 1
    return final_interesting_r


def __get_max_candidate_rules_count(interesting_rules, max_count):
    if not max_count:
        return interesting_rules
    else:
        total_interesting_rules_count = len(interesting_rules)
        count = max_count if total_interesting_rules_count >= max_count else total_interesting_rules_count
        LOGGER.debug('returning %d candidate rules out of %d', count, total_interesting_rules_count)
        return interesting_rules[:count]


def __generate_initial_candidates(dataset, operators):
    heads = list(dataset.columns.values)
    cand_rules = list([__create_binary_rule(item1, item2, op)
                       for item1, item2 in combinations(heads,2)
                       for op in operators if op.operator_type != OperatorType.NE and op.support_features(item1, item2)

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
    total_rules_count = len(candidate_rules)
    total_checked_rules=0
    LOGGER.info('mining interesting rules for %d rules',total_rules_count)
    interesting_rules = []
    for rule in candidate_rules:
        total_checked_rules += 1
        s,c, m = __calculate_support_confidence_membership(rule, dataset, min_membership)
        if s >= min_s and c >= min_c:
            interesting_rules.append((rule,m))
            LOGGER.debug('grule %s support  %f confidence %f membership %f' % (str(rule), s, c, m))
        LOGGER.debug('check %d/%d rules fot interesting ', total_checked_rules, total_rules_count)
    return interesting_rules


def __calculate_support_confidence_membership(rule:GRule, dataset, min_membership):
    s_counter =0
    c_counter=0
    m_c=0
    n = dataset.shape[0]
    for index,row in dataset.iterrows():
        if __all_rule_items_present(row, rule.get_all_items()):
            s_counter+=1
            membership_degree = rule.calculate_membership_degree_avg(row)
            if membership_degree >= min_membership:
                c_counter+=1
                m_c += membership_degree
    s_avg = s_counter/n
    c_avg = c_counter/n
    m_avg = m_c/n
    #print('grule %s support  %f confidence %f membership %f' % (str(rule), s_avg, c_avg, m_avg))
    return s_avg, c_avg, m_avg


def __all_rule_items_present(data_row, items):
    for item in items:
        if item.category not in data_row:
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
            if None not in terms and _is_terms_operands_compatible(terms):
                new_terms = gRule.join(join_rule, *terms)
                new_grule = GRule(new_length)
                new_grule.add_terms(new_terms)
                new_candidate.append(new_grule)
                LOGGER.debug('grule : %s , grule : %s joint by [%s] to new grule : %s '
                             ,str(rule1), str(rule2),join_rule, str(new_grule))
            else:
                LOGGER.debug('not marble grule : %s and grule : %s ', str(rule1), str(rule2))

    return new_candidate


def _is_terms_operands_compatible(*terms):
    if len(*terms) <= 1 or None in terms:
        return False
    for index in range(len(terms)-1):
        current_term_item_identifier = terms[index].item.get_identifier()
        current_term_operator = terms[index].operator
        next_term_item_identifier = terms[index+1].item.get_identifier()
        if not current_term_operator.support_features(current_term_item_identifier, next_term_item_identifier):
            LOGGER.debug('terms %s operands not compatible : ' % str(terms))
            return False
    return True


def create_grar_object(grule : GRule, membership):
    obj = {
        "rule_terms": grule.create_object(),
        "membership": membership
    }
    return obj

def build_from_obj(obj):
    return GRule.build_from_obj(obj), obj.get('membership')

def _debug(r1:GRule, r2:GRule):
    for r1t in r1.get_terms():
        if r1t.item not in r2.get_all_items():
            return True
    return False


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

