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
import numpy as np

LOGGER = logging.getLogger(__name__)
HYGRAR_PERSISTENCE= filesystem.get_relative_to_home('hygrar')


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

    def __init__( self, run_id, use_case, grar_min_support, grar_min_confidence, grar_min_membership_degree, nn_model_creation='retrain', rule_max_length=None , d1_percentage=0 ):
        self.min_support = grar_min_support
        self.min_confidence = grar_min_confidence
        self.min_membership_degree = grar_min_membership_degree
        self.rule_max_length=rule_max_length
        self.nn_model_creation_strategy = nn_model_creation
        self.use_case = use_case
        self.run_id = run_id
        self.training_phase1_percenatge = d1_percentage

    @Timer(text="hgrar training executed in {:.2f} seconds")
    def train(self,x: pd.DataFrame,y: pd.DataFrame, use_nn_types= ('ALL'), balanced_buckets=True,merge_buckets = False):
        LOGGER.info('========================================================================')
        LOGGER.info('========================================================================')
        LOGGER.info('hygrar training started ')
        LOGGER.info('========================================================================')
        LOGGER.info('========================================================================')
        self.features_col = list(x.columns.values)
        self.class_col = y.columns.values[0]
        (d1_x, d1_y) , (d2_x, d2_y) = self.split_to_d1_d2(x, y)
        LOGGER.info('hygrar training create/load NN models ')
        nn_path = filesystem.join(HYGRAR_PERSISTENCE,self.run_id,self.use_case)
        ann_models = ann_creator.create_ANN_models(nn_path, self.use_case, pd.concat([d1_x, d1_y], axis=1), self.features_col, self.class_col,
                                                   use_nn_types=use_nn_types,balanced_buckets=balanced_buckets,merge_buckets=merge_buckets,
                                                   nn_model_strategy=self.nn_model_creation_strategy,
                                                   perceptron_init_param= self.PERCEPTRON_INIT_PARAM,
                                                   rfbn_init_param=self.PFBN_INIT_PARAM)
        operators = []
        for m in ann_models:
            operators.append(AnnOperator(OperatorType.FAULTY, m))
            operators.append(AnnOperator(OperatorType.NON_FAULTY, m))
        grar_subset = util.create_uniqe_classes_subsets(util.concat_datasets([d2_x,d2_y],axis=1), self.class_col)
        LOGGER.info('hygrar training create non faulty detector grars ')
        self.interesting_grar_not_faulty = grar.start(grar_subset['False'].drop(columns=[self.class_col]), operators,
                                                  self.min_support,self.min_confidence,self.min_membership_degree,
                                                      rule_max_length=self.rule_max_length, max_rule_count=None)
        self.not_faulty_rule_actual_length = self.interesting_grar_not_faulty[0][0].length
        LOGGER.info('hygrar training create faulty detector grars ')
        self.interesting_grar_faulty = grar.start(grar_subset['True'].drop(columns=[self.class_col]), operators,
                                              self.min_support,self.min_confidence,self.min_membership_degree,
                                                  rule_max_length=self.rule_max_length, max_rule_count=None)
        self.faulty_rule_actual_length = self.interesting_grar_faulty[0][0].length

    @Timer(text="hgrar extend_training executed in {:.2f} seconds")
    def extend_training( self, x_not_extended: pd.DataFrame, y_not_extended: pd.DataFrame,
                         x_extended: pd.DataFrame, y_extended: pd.DataFrame, extended_rules_max_length,
                         use_nn_types=('ALL'), balanced_buckets=True,
               merge_buckets=False , extend_old_grars_training=True ):
        self.rule_max_length = extended_rules_max_length
        extended_features_col = list([ext_col for ext_col in x_extended.columns.values if ext_col not in self.features_col])
        (d1_x, d1_y), (d2_x, d2_y) = self.split_to_d1_d2(x_extended, y_extended)
        nn_path = filesystem.join(HYGRAR_PERSISTENCE, self.run_id, self.use_case)
        if extend_old_grars_training:
            self._examine_old_grars(x_not_extended, y_not_extended,(d1_x, d1_y), (d2_x, d2_y),nn_path, balanced_buckets)

        extended_ann_models = ann_creator.create_ANN_models(nn_path, self.use_case, pd.concat([d1_x, d1_y], axis=1),
                                                   self.features_col + extended_features_col, self.class_col,
                                                   use_nn_types=use_nn_types, balanced_buckets=balanced_buckets,
                                                   merge_buckets=merge_buckets,
                                                   nn_model_strategy=self.nn_model_creation_strategy,
                                                   perceptron_init_param=self.PERCEPTRON_INIT_PARAM,
                                                   rfbn_init_param=self.PFBN_INIT_PARAM)
        extended_operators = []
        for m in extended_ann_models:
            extended_operators.append(AnnOperator(OperatorType.FAULTY, m))
            extended_operators.append(AnnOperator(OperatorType.NON_FAULTY, m))

        extended_d2 = util.create_uniqe_classes_subsets(util.concat_datasets([d2_x, d2_y], axis=1), self.class_col)

        LOGGER.info('hygrar training extend non faulty detector grars ')
        self.interesting_grar_not_faulty = grar.extend_grars(extended_d2['False'].drop(columns=[self.class_col]),self.interesting_grar_not_faulty, self.features_col,
                                                             extended_features_col,extended_operators, self.not_faulty_rule_actual_length,
                                                             self.min_support,self.min_confidence,self.min_membership_degree,
                                                             rule_max_length=extended_rules_max_length, max_rule_count=None)

        self.not_faulty_rule_actual_length = self.interesting_grar_not_faulty[0][0].length

        LOGGER.info('hygrar training extend faulty detector grars ')
        self.interesting_grar_faulty = grar.extend_grars(extended_d2['True'].drop(columns=[self.class_col]),self.interesting_grar_faulty, self.features_col,
                                                             extended_features_col,extended_operators, self.faulty_rule_actual_length,
                                                             self.min_support,self.min_confidence,self.min_membership_degree,
                                                             rule_max_length=extended_rules_max_length, max_rule_count=None)
        self.faulty_rule_actual_length = self.interesting_grar_faulty[0][0].length
        self.features_col = self.features_col + extended_features_col


    def _examine_old_grars(self,x_not_extended: pd.DataFrame, y_not_extended: pd.DataFrame,
                         extended_d1, extended_d2 ,nn_path, balanced_buckets):

        (d1_x, d1_y), (d2_x, d2_y) = extended_d1, extended_d2
        ann_creator.extend_ANN_models_training(nn_path, pd.concat([d1_x, d1_y], axis=1), self.features_col,
                                               self.class_col, balanced_buckets=balanced_buckets)
        all_x = util.concat_datasets([x_not_extended, d2_x])
        all_y = util.concat_datasets([y_not_extended, d2_y])
        grar_subset = util.create_uniqe_classes_subsets(util.concat_datasets([all_x, all_y], axis=1), self.class_col)
        self.interesting_grar_not_faulty = grar.rexamin_grars_membership(grar_subset['Fales'],
                                                                         self.interesting_grar_not_faulty,
                                                                         self.min_support, self.min_confidence,
                                                                         self.min_membership_degree)
        self.not_faulty_rule_actual_length = self.interesting_grar_not_faulty[0][0].length

        self.interesting_grar_faulty = grar.rexamin_grars_membership(grar_subset['True'],
                                                                     self.interesting_grar_faulty,
                                                                     self.min_support, self.min_confidence,
                                                                     self.min_membership_degree)
        self.faulty_rule_actual_length = self.interesting_grar_faulty[0][0].length

    def split_to_d1_d2( self,x, y):
        if self.training_phase1_percenatge == 0:
            return (x,y), (x,y)
        else:
            true_cond = y[self.class_col] == True
            false_cond = y[self.class_col] == False

            x_df_true = x[true_cond]
            x_df_false = x[false_cond]

            y_df_true = y[true_cond]
            y_df_false = y[false_cond]

            true_df_sample = y_df_true.sample(frac=self.training_phase1_percenatge , replace=False).index
            false_df_sample = y_df_false.sample(frac=self.training_phase1_percenatge, replace=False).index

            ph1_x_true = x_df_true[x_df_true.index.isin(true_df_sample.values)]
            ph1_x_false = x_df_false[x_df_false.index.isin(false_df_sample.values)]

            ph1_y_true = y_df_true[y_df_true.index.isin(true_df_sample.values)]
            ph1_y_false = y_df_false[y_df_false.index.isin(false_df_sample.values)]

            ph2_x_true = x_df_true[~x_df_true.index.isin(true_df_sample.values)]
            ph2_x_false = x_df_false[~y_df_false.index.isin(false_df_sample.values)]

            ph2_y_true = y_df_true[~y_df_true.index.isin(true_df_sample.values)]
            ph2_y_false = y_df_false[~y_df_false.index.isin(false_df_sample.values)]

            return (pd.concat([ph1_x_true, ph1_x_false]), pd.concat([ph1_y_true, ph1_y_false])),\
                   (pd.concat([ph2_x_true, ph2_x_false]), pd.concat([ph2_y_true, ph2_y_false]))

    @Timer(text="hgrar predict executed in {:.2f} seconds")
    def predict(self, dataset :pd.DataFrame, grar_count, bulk=False):
        faulty_grar_count = grar_count if len(self.interesting_grar_faulty) >= grar_count else len(self.interesting_grar_faulty)
        non_faulty_grar_count = grar_count if len(self.interesting_grar_not_faulty) >= grar_count else len(self.interesting_grar_not_faulty)
        sorted_faulty_grar = sorted(self.interesting_grar_faulty, key=lambda gr: gr[1], reverse=True)[0:faulty_grar_count]
        LOGGER.debug('\nselected faulty grars \n%s\n', print_grars(sorted_faulty_grar))
        sorted_non_faulty_grar = sorted(self.interesting_grar_not_faulty, key=lambda gr: gr[1], reverse=True)[0:non_faulty_grar_count]
        LOGGER.debug('\nselected non faulty grars \n%s\n', print_grars(sorted_non_faulty_grar))
        predictions = []
        if not bulk:
            for _, row in dataset.iterrows():
                LOGGER.debug('\n\n data row %s ', str(row.values))
                LOGGER.debug('---apply faulty grars---')
                faulty_dist = _calculate_diff(row, sorted_faulty_grar)
                LOGGER.debug('---apply non faulty grars---')
                nonfaulty_dist = _calculate_diff(row, sorted_non_faulty_grar)
                faulty = True if faulty_dist < nonfaulty_dist else False
                r_obj = matrix.create_prediction_obj( row_data = row.values,row_data_columns= dataset.columns.values,
                                                      true_class = row[self.class_col], prediction=faulty )
                predictions.append(r_obj)
                if faulty != row[self.class_col]:
                    LOGGER.debug('wrong prediction')
            return predictions
        else:
            LOGGER.debug('---apply Bulk faulty grars---')
            faulty_distances = _calculate_diff_bulk(dataset, sorted_faulty_grar)
            LOGGER.debug('---apply Bulk non faulty grars---')
            nonfaulty_distances = _calculate_diff_bulk(dataset, sorted_non_faulty_grar)
            for index, row in dataset.iterrows():
                f_dist = faulty_distances[index]
                nf_dist = nonfaulty_distances[index]
                faulty = True if f_dist < nf_dist else False
                r_obj = matrix.create_prediction_obj(row_data=row.values, row_data_columns=dataset.columns.values,
                                                     true_class=row[self.class_col], prediction=faulty)
                predictions.append(r_obj)
                # if faulty != row[self.class_col]:
                #     LOGGER.debug('\n\n data row %s ', str(row.values))
                #     logging.debug('Faulty grar distance = %f', f_dist)
                #     logging.debug('Non Faulty grar distance = %f', nf_dist)
                #     LOGGER.debug('wrong prediction')
            return predictions

    def save_grars( self ):
        obj = {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_membership_degree": self.min_membership_degree,
            "rule_max_length": self.rule_max_length,
            "nn_model_creation_strategy": self.nn_model_creation_strategy,
            "run_id": self.run_id,
            "use_case" : self.use_case,
            "features_col": self.features_col,
            "class_col": self.class_col,
            "training_phase1_percenatge": self.training_phase1_percenatge,
            "not_faulty_rule_actual_length":self.not_faulty_rule_actual_length,
            "faulty_rule_actual_length":self.faulty_rule_actual_length,
            "faulty_grar" : list([ grar.create_grar_object(grule,membership) for grule, membership in self.interesting_grar_faulty]),
            "non_faulty_grar": list(
                [grar.create_grar_object(grule, membership) for grule, membership in self.interesting_grar_not_faulty])
        }
        file_path = filesystem.join(HYGRAR_PERSISTENCE,self.run_id, self.use_case )
        filesystem.create_path(file_path,exist_ok=True)
        with open(filesystem.join(file_path,'grars.json'), 'w') as hygrar_file:
            json.dump(obj, hygrar_file, indent=4)


    def load_hgrar(run_id,use_case, grar_count=None):
        file_path = filesystem.join(HYGRAR_PERSISTENCE,run_id, use_case )
        obj={}
        with open(filesystem.join(file_path, 'grars.json'), 'r') as hygrar_file:
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
        d1_percentage = object_dict.get('training_phase1_percenatge')
        not_faulty_rule_actual_length = object_dict.get("not_faulty_rule_actual_length")
        faulty_rule_actual_length = object_dict.get("faulty_rule_actual_length")
        hgrar = HyGRAR(run_id, min_support, min_confidence, min_membership_degree, nn_model_creation_strategy, rule_max_length=rule_max_length, d1_percentage=d1_percentage)
        hgrar.interesting_grar_not_faulty = []
        hgrar.interesting_grar_faulty = []
        hgrar.features_col = features_col
        hgrar.class_col = class_col
        hgrar.not_faulty_rule_actual_length = not_faulty_rule_actual_length
        hgrar.faulty_rule_actual_length = faulty_rule_actual_length
        faulty_grar_objects = sorted(object_dict.get('faulty_grar'), key= lambda x:x['membership'], reverse=True)
        non_faulty_grar_objects = sorted(object_dict.get('non_faulty_grar'), key= lambda x:x['membership'], reverse=True)

        if grar_count:
            faulty_grar_objects = faulty_grar_objects[:grar_count]
            non_faulty_grar_objects = non_faulty_grar_objects[:grar_count]

        for faulty_grar in faulty_grar_objects:
            hgrar.interesting_grar_faulty.append(grar.build_from_obj(faulty_grar))
        for non_faulty_grar in non_faulty_grar_objects:
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

def _calculate_diff_bulk(dataset, grar:[]):
    total_rules_diff=0
    n = len(grar)
    for r, m in grar:
        m_arr = np.full(fill_value=m, shape=(len(dataset),))
        mr = r.calculate_membership_degree_bulk(dataset)
        total_rules_diff += abs(m_arr - mr)
    #LOGGER.debug('grar (%s , %f) membership for data is %f'% (str(r),m, mr))
    avg_diff = total_rules_diff / n
    return avg_diff

def print_grars(grars):
    print_str=''
    for grul, m in grars:
        print_str+='grule %s with membership %f \n' % (str(grul), m)
    return print_str

