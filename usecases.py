HYGRAR_TEST_USECASES = [
    {
        'usecase_name': 'ar1',
        'test_datasets': ['ar1.csv'],
        'test_stages': [
            {
                'stage_name': 'stage_1',
                'training_datasets': ['ar3.csv', 'ar4.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            },
            {
                'stage_name': 'stage_2',
                'training_datasets': ['ar3.csv', 'ar4.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'halstead_vocabulary', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            }

        ],
        'dataset_transformation': None
    },
    {
        'usecase_name': 'ar3',
        'test_datasets': ['ar3.csv'],
        'test_stages': [
            {
                'stage_name': 'stage_1',
                'training_datasets': ['ar1.csv', 'ar4.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            },
            {
                'stage_name': 'stage_2',
                'training_datasets': ['ar1.csv', 'ar4.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'halstead_vocabulary', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            }

        ],
        'dataset_transformation': None
    },
    {
        'usecase_name': 'ar4',
        'test_datasets': ['ar4.csv'],
        'test_stages': [
            {
                'stage_name': 'stage_1',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            },
            {
                'stage_name': 'stage_2',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar5.csv', 'ar6.csv'],
                'features': ['unique_operators', 'halstead_vocabulary', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            }

        ],
        'dataset_transformation': None
    },
    {
        'usecase_name': 'ar5',
        'test_datasets': ['ar5.csv'],
        'test_stages': [
            {
                'stage_name': 'stage_1',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar4.csv', 'ar6.csv'],
                'features': ['unique_operators', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            },
            {
                'stage_name': 'stage_2',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar4.csv', 'ar6.csv'],
                'features': ['unique_operators', 'halstead_vocabulary', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            }

        ],
        'dataset_transformation': None
    },
    {
        'usecase_name': 'ar6',
        'test_datasets': ['ar6.csv'],
        'test_stages': [
            {
                'stage_name': 'stage_1',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar4.csv', 'ar5.csv'],
                'features': ['unique_operators', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            },
            {
                'stage_name': 'stage_2',
                'training_datasets': ['ar1.csv', 'ar3.csv','ar4.csv', 'ar5.csv'],
                'features': ['unique_operators', 'halstead_vocabulary', 'unique_operands'],
                'class': 'defects',
                'result_file_name_prefix': 'AR'
            }

        ],
        'dataset_transformation': None
    }
]

ADAPTIV_GRAR_TEST_USECASES =[]

def get_usecases(names):
    ret_usecases = []
    for uc in HYGRAR_TEST_USECASES:
        if uc.get('usecase_name') in names:
            ret_usecases.append(uc)
    return ret_usecases

