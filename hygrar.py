import GRAR as grar
from utils import datapartitional as util
from model.grar.perceptron_operator import AnnOperator
from NR import ann_creator
from model.grar.operator import OperatorType
import pandas as pd
from utils import matrix, filesystem
from utils.timer import Timer
import logging
import json

LOGGER = logging.getLogger(__name__)
HYGRAR_PERSISTENCE='hygrar_persistence'


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

    @Timer(text="hgrar training executed in {:.2f} seconds")
    def train(self,x: pd.DataFrame,y: pd.DataFrame):
        LOGGER.info('hygrar training started ')
        self.features_col = x.columns
        self.class_col = y.name
        LOGGER.info('hygrar training create/load NN models ')
        ann_models = ann_creator.create_ANN_models(self.run_id, pd.concat([x,y],axis=1) , self.features_col, self.class_col,
                                                   nn_model_strategy=self.nn_model_creation_strategy,
                                                   perceptron_init_param= self.PERCEPTRON_INIT_PARAM,
                                                   rfbn_init_param=self.PFBN_INIT_PARAM )
        operators = []
        for m in ann_models:
            operators.append(AnnOperator(OperatorType.FAULTY, m))
            operators.append(AnnOperator(OperatorType.NON_FAULTY, m))
        grar_subset = util.create_uniqe_classes_subsets(pd.concat([x,y],axis=1), self.class_col)
        LOGGER.info('hygrar training create non faulty detector grars ')
        self.interesting_grar_not_faulty = grar.start(grar_subset['False'].drop(columns=[self.class_col]), operators,
                                                  self.min_support,self.min_confidence,self.min_membership_degree,rule_max_length=self.rule_max_length)
        LOGGER.info('hygrar training create faulty detector grars ')
        self.interesting_grar_faulty = grar.start(grar_subset['True'].drop(columns=[self.class_col]), operators,
                                              self.min_support,self.min_confidence,self.min_membership_degree, rule_max_length=self.rule_max_length)

    def train2(self,datasets, features_col, class_col):
        self.features_col = features_col
        self.class_col = class_col
        ann_models = ann_creator.create_ANN_models2(self.run_id, datasets , self.features_col, self.class_col,
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

        grar_subset = util.create_uniqe_classes_subsets(util.concat_datasets(datasets)[self.features_col + [ self.class_col]], self.class_col)
        self.interesting_grar_not_faulty = grar.start(grar_subset['False'].drop(columns=[self.class_col]), all,
                                                  self.min_support,self.min_confidence,self.min_membership_degree,2)

        self.interesting_grar_faulty = grar.start(grar_subset['True'].drop(columns=[self.class_col]), all,
                                              self.min_support,self.min_confidence,self.min_membership_degree, 2)

    @Timer(text="hgrar predict executed in {:.2f} seconds")
    def predict(self, dataset :pd.DataFrame, grar_count):
        faulty_grar_count = grar_count if len(self.interesting_grar_faulty) >= grar_count else len(self.interesting_grar_faulty)
        non_faulty_grar_count = grar_count if len(self.interesting_grar_not_faulty) >= grar_count else len(self.interesting_grar_not_faulty)
        sorted_faulty_grar = sorted(self.interesting_grar_faulty, key=lambda gr: gr[1], reverse=True)[0:faulty_grar_count]
        LOGGER.debug('\nselected faulty grars %s\n', print_grars(sorted_faulty_grar))
        sorted_non_faulty_grar = sorted(self.interesting_grar_not_faulty, key=lambda gr: gr[1], reverse=True)[0:non_faulty_grar_count]
        LOGGER.debug('\nselected non faulty grars %s\n', print_grars(sorted_non_faulty_grar))
        predictions = []
        for _, row in dataset.iterrows():
            LOGGER.debug('\n\n data row %s', str(row.values))
            LOGGER.debug('---apply faulty grars---')
            faulty_dist = _calculate_diff(row, sorted_faulty_grar)
            LOGGER.debug('---apply non faulty grars---')
            nonfaulty_dist = _calculate_diff(row, sorted_non_faulty_grar)
            faulty = True if faulty_dist < nonfaulty_dist else False
            r_obj = matrix.create_prediction_obj( row_data = row.values, true_class = row[self.class_col], prediction=faulty )
            predictions.append(r_obj)
            if faulty != row[self.class_col]:
                LOGGER.debug('wrong prediction')
        return predictions

    def save_grars( self ):
        obj = {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_membership_degree": self.min_membership_degree,
            "rule_max_length": self.rule_max_length,
            "nn_model_creation_strategy": self.nn_model_creation_strategy,
            "run_id": self.run_id,
            "features_col": self.features_col,
            "class_col": self.class_col,
            "faulty_grar" : list([ grar.create_grar_object(grule,membership) for grule, membership in self.interesting_grar_faulty]),
            "non_faulty_grar": list(
                [grar.create_grar_object(grule, membership) for grule, membership in self.interesting_grar_not_faulty])
        }
        file_path = filesystem.get_relative_to_home(HYGRAR_PERSISTENCE)
        filesystem.create_path(file_path,exist_ok=True)
        with open(filesystem.join(file_path,'grars_'+self.run_id+'.json'), 'w') as hygrar_file:
            json.dump(obj, hygrar_file, indent=4)


    def load_hgrar(run_id, grar_count=None):
        file_path = filesystem.get_relative_to_home(HYGRAR_PERSISTENCE)
        obj={}
        with open(filesystem.join(file_path, 'grars_' + run_id + '.json'), 'r') as hygrar_file:
            obj = json.load(hygrar_file)
        return  HyGRAR.create_from_obj(obj, grar_count)


    def create_from_obj(object_dict,grar_count):
        min_support = object_dict.get('min_support')
        min_confidence = object_dict.get('min_confidence')
        min_membership_degree = object_dict.get('min_membership_degree')
        rule_max_length = object_dict.get('rule_max_length')
        nn_model_creation_strategy = object_dict.get('nn_model_creation_strategy')
        features_col = object_dict.get("features_col")
        class_col = object_dict.get("class_col")

        run_id = object_dict.get('run_id')
        hgrar = HyGRAR(run_id, min_support, min_confidence, min_membership_degree, nn_model_creation_strategy, rule_max_length=rule_max_length)
        hgrar.interesting_grar_not_faulty = []
        hgrar.interesting_grar_faulty = []
        hgrar.features_col = features_col
        hgrar.class_col = class_col

        faulty_grar_objexts = sorted(object_dict.get('faulty_grar'), key= lambda x:x['membership'], reverse=True)
        non_faulty_grar_objexts = sorted(object_dict.get('non_faulty_grar'), key= lambda x:x['membership'], reverse=True)

        if grar_count:
            faulty_grar_objexts = faulty_grar_objexts[:grar_count]
            non_faulty_grar_objexts = faulty_grar_objexts[:grar_count]

        for faulty_grar in faulty_grar_objexts:
            hgrar.interesting_grar_faulty.append(grar.build_from_obj(faulty_grar))
        for non_faulty_grar in non_faulty_grar_objexts:
            hgrar.interesting_grar_not_faulty.append(grar.build_from_obj(non_faulty_grar))
        return hgrar





def _calculate_diff(data_row, grar: []):
    total_rules_diff=0
    n = len(grar)
    for r, m in grar:
        mr = r.calculate_membership_degree(data_row)
        total_rules_diff += abs(m - mr)
        LOGGER.debug('grar (%s , %f) membership for data is %f'% (str(r),m, mr))
    avg_diff = total_rules_diff / n
    LOGGER.debug('final grars diff is %f'% avg_diff)
    return avg_diff

def print_grars(grars):
    print_str=''
    for grul, m in grars:
        print_str+='grule %s with membership %f \n' % (str(grul), m)
    return print_str

