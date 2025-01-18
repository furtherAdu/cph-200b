from collections import defaultdict

feature_config = dict(
    faisalabad=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking'],
    unos=["init_age", "gender", "hgt_cm_tcr", "wgt_kg_tcr", "diab", "ventilator_tcr",
          "ecmo_tcr", "most_rcnt_creat", "abo_A", "abo_B", "abo_O", "vad_while_listed",
          "iabp_tcr", "init_bmi_calc", "tah", "inotropic"]
)

clinical_feature_type = dict(
    faisalabad={
        'age': 'numerical',
        'anaemia': 'categorical',
        'creatinine_phosphokinase': 'numerical',
        'diabetes': 'categorical',
        'ejection_fraction': 'numerical',
        'high_blood_pressure': 'categorical',
        'platelets': 'numerical',
        'serum_creatinine': 'numerical',
        'serum_sodium': 'numerical',
        'sex': 'categorical',
        'smoking': 'categorical',
    },
    unos={'init_age': 'numerical',
          'gender': 'categorical',
          'hgt_cm_tcr': 'numerical',
          'wgt_kg_tcr': 'numerical',
          'diab': 'categorical',
          'ventilator_tcr': 'categorical',
          'ecmo_tcr': 'categorical',
          'most_rcnt_creat': 'numerical',
          'abo_A': 'categorical',
          'abo_B': 'categorical',
          'abo_O': 'categorical',
          'vad_while_listed': 'categorical',
          'days_stat1': 'numerical',
          'days_stat1a': 'numerical',
          'days_stat2': 'numerical',
          'days_stat1b': 'numerical',
          'iabp_tcr': 'categorical',
          'init_bmi_calc': 'numerical',
          'tah': 'categorical',
          'inotropic': 'categorical'}
)

# subgroup_feature_type = dict(
#     age='histogram',
#     sex='categorical'
# )

dataset_stats = {'unos': {'init_age': {'mean': 44.312182195803196, 'std': 19.00969775184649},
                          'hgt_cm_tcr': {'mean': 164.42901203179832, 'std': 32.519173421696735},
                          'wgt_kg_tcr': {'mean': 75.5248756320275, 'std': 28.273276335328312},
                          'most_rcnt_creat': {'mean': 1.3746322423547948, 'std': 1.1524063830664726},
                          'init_bmi_calc': {'mean': 26.068502282660212, 'std': 8.497129255043367}},
                 'faisalabad': {'age': {'mean': 60.83389297658862, 'std': 11.894809074044478},
                                'creatinine_phosphokinase': {'mean': 581.8394648829432,
                                                             'std': 970.2878807124363},
                                'ejection_fraction': {'mean': 38.08361204013378, 'std': 11.834840741039173},
                                'platelets': {'mean': 263358.02926421404, 'std': 97804.23686859828},
                                'serum_creatinine': {'mean': 1.3938795986622072, 'std': 1.034510064089853},
                                'serum_sodium': {'mean': 136.62541806020067, 'std': 4.412477283909233}}}

feature_transforms = defaultdict(lambda: defaultdict(list))

for dataset_name, feature_dict in clinical_feature_type.items():
    for feature_name, feature_type in feature_dict.items():
        feature_transforms[dataset_name][feature_type].append(feature_name)
