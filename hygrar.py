import GRAR as grar
from utils import datapartitional as util
from model.grar.perceptron_operator import AnnOperator
from NR import ann_creator
from model.grar.operator import OperatorType
import pandas as pd
from utils import matrix
from utils.timer import Timer


class HyGRAR:
    PERCEPTRON_INIT_PARAM = {
        'learning_rate': 0.1,
        'input_shape': (2,),
        'batch_size': 10,
        'epochs': 400,
        'loss': 'mean_squared_error',
        'early_stop_patience_ratio':0.5,
        'early_stop_monitor_metric':'val_loss'
    }

    PFBN_INIT_PARAM = {
        'betas': 0.5,
        'centers': 2,
        'input_shape': (2,),
        'batch_size': 10,
        'epochs': 400,
        'loss': 'mean_squared_error',
        'early_stop_patience_ratio': 0.5,
        'early_stop_monitor_metric': 'val_loss'
    }


    def __init__(self, run_id, grar_min_support, grar_min_confidence, grar_min_membership_degree,nn_model_creation='retrain', rule_max_length=None):
        self.min_support = grar_min_support
        self.min_confidence = grar_min_confidence
        self.min_membership_degree = grar_min_membership_degree
        self.rule_max_length=rule_max_length
        self.nn_model_creation_strategy = nn_model_creation
        self.run_id = run_id

    def train(self,x: pd.DataFrame,y: pd.DataFrame):
        self.features_col = x.columns
        self.class_col = y.name
        ann_models = ann_creator.create_ANN_models(self.run_id, pd.concat([x,y],axis=1) , self.features_col, self.class_col,
                                                   nn_model_strategy=self.nn_model_creation_strategy,
                                                   perceptron_init_param= self.PERCEPTRON_INIT_PARAM,
                                                   rfbn_init_param=self.PFBN_INIT_PARAM )
        operators_faulty = []
        operators_non_faulty = []
        all = []
        for m in ann_models:
            operators_faulty.append(AnnOperator(OperatorType.FAULTY, m))
            operators_non_faulty.append(AnnOperator(OperatorType.NON_FAULTY, m))

        all.extend(operators_faulty)
        all.extend(operators_non_faulty)

        grar_subset = util.create_uniqe_classes_subsets(pd.concat([x,y],axis=1), self.class_col)
        self.interesting_grar_not_faulty = grar.start(grar_subset['False'].drop(columns=[self.class_col]), all,
                                                  self.min_support,self.min_confidence,self.min_membership_degree,2)

        self.interesting_grar_faulty = grar.start(grar_subset['True'].drop(columns=[self.class_col]), all,
                                              self.min_support,self.min_confidence,self.min_membership_degree, 2)

    @Timer(text="hgrar predict executed in {:.2f} seconds")
    def predict(self, dataset :pd.DataFrame, grar_count):
        faulty_grar_count = grar_count if len(self.interesting_grar_faulty) >= grar_count else len(self.interesting_grar_faulty)
        non_faulty_grar_count = grar_count if len(self.interesting_grar_not_faulty) >= grar_count else len(self.interesting_grar_not_faulty)
        sorted_faulty_grar = sorted(self.interesting_grar_faulty, key=lambda gr: gr[1], reverse=True)[0:faulty_grar_count]
        print('selected faulty grars \n', print_grars(sorted_faulty_grar))
        sorted_non_faulty_grar = sorted(self.interesting_grar_not_faulty, key=lambda gr: gr[1], reverse=True)[0:non_faulty_grar_count]
        print('selected non faulty grars \n', print_grars(sorted_non_faulty_grar))
        predictions = []
        for _, row in dataset.iterrows():
            print('\n\n data row %s', str(row.values))
            print('---apply faulty grars---')
            faulty_dist = _calculate_diff(row, sorted_faulty_grar)
            print('---apply non faulty grars---')
            nonfaulty_dist = _calculate_diff(row, sorted_non_faulty_grar)
            faulty = True if faulty_dist < nonfaulty_dist else False
            r_obj = matrix.create_prediction_obj( row_data = row.values, true_class = row[self.class_col], prediction=faulty )
            predictions.append(r_obj)
            if faulty != row[self.class_col]:
                print('wrong prediction')
        return predictions

def _calculate_diff(data_row, grar: []):
    total_rules_diff=0
    n = len(grar)
    for r, m in grar:
        mr = r.calculate_membership_degree(data_row)
        total_rules_diff += abs(m - mr)
        print('grar (%s , %f) membership for data is %f'% (str(r),m, mr))
    avg_diff = total_rules_diff / n
    print('final grars diff is %f'% avg_diff)
    return avg_diff

def print_grars(grars):
    print_str=''
    for grul, m in grars:
        print_str+='grule %s with membership %f \n' % (str(grul), m)
    return print_str

