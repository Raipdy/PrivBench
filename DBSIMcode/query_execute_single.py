import numpy as np
import json
import pandas as pd
import argparse


import common

def get_qerror(pred, label):
    qerror = np.maximum(pred/label, label/pred)
    return qerror

def MakeTable(dataset):

    if dataset == 'dmv':
        table = LoadDmv()
    elif dataset == 'census':
        table = LoadCensus()

    return table, None

def LoadCensus(filename_or_df='census.csv'):
    if isinstance(filename_or_df, str):
        filename_or_df = '/home/raipdy/SAM/sam_single/datasets/{}'.format(filename_or_df)
    else:
        assert (isinstance(filename_or_df, pd.DataFrame))
    cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    type_casts = {}
    return common.CsvTable('Census', filename_or_df, cols, type_casts, header=None)

def LoadDmv(filename='DMV.csv'):
    csv_file = '/home/raipdy/SAM/sam_single/datasets/{}'.format(filename)
    cols = [
        'Record Type','Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def load_dataset(dataset, file):
    with open(file, 'r', encoding="utf8") as f:
        workload_stats = json.load(f)
    card_list_tmp = workload_stats['card_list']
    query_list = workload_stats['query_list']

    table, _ = MakeTable(dataset)

    card_list_tmp = [float(card)/table.cardinality for card in card_list_tmp]

    columns_list = []
    operators_list = []
    vals_list = []
    card_list = []
    for i, query in enumerate(query_list):
        if card_list_tmp[i] > 0:
            cols = query[0]
            ops = query[1]
            vals = query[2]
            columns_list.append(cols)
            operators_list.append(ops)
            vals_list.append(vals)
            card_list.append(card_list_tmp[i])
    return {"column": columns_list, "operator": operators_list, "val": vals_list, "card": card_list}

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


def Query(table, columns, operators, vals, return_masks=False, return_crad_and_masks=False):
    assert len(columns) == len(operators) == len(vals)
    bools = None
    # print(table.name)
    for c, o, v in zip(columns, operators, vals):
        if table.name in ['DMV', 'Census']:
            inds = [False] * table.cardinality
            inds = np.array(inds)
            is_nan = pd.isnull(c.data)
            if np.any(is_nan):
                v = np.array(v).astype(c.data.dtype)
                # print(type(v))
                # print(v)
                inds[~is_nan] = OPS[o](c.data[~is_nan], v)
            else:
                v = np.array(v).astype(c.data.dtype)
                inds = OPS[o](c.data, v)
        else:
            v = np.array(v).astype(c.data.dtype)
            inds = OPS[o](c.data, v)

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    if return_masks:
        return bools
    elif return_crad_and_masks:
        return c, bools
    return c

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dmv', help='Dataset.')
    parser.add_argument('--data-file', type=str, default='../../result/test/DMV.csv', help='Generated data file')
    parser.add_argument('--query-file', type=str, default='../../queries/dmv_test.txt', help='Query file')

    args = parser.parse_args()

    train_data_raw = load_dataset(args.dataset, args.query_file)

    start_idx = 0
    test_num = len(train_data_raw['val'])

    if args.dataset == "census":
        sample_frame = pd.read_csv(args.data_file, header=0, names=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        sample_frame = sample_frame.astype({0:'float64', 4:'float64', 10:'float64', 11:'float64', 12:'float64'})
        sample_table = LoadCensus(sample_frame)
    elif args.dataset == "dmv":
        csv_file = args.data_file

        cols = [
                'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
                'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
                'Suspension Indicator', 'Revocation Indicator'
            ]
        type_casts = {'Reg Valid Date': np.datetime64}
        sample_table = common.CsvTable('DMV', csv_file, cols, type_casts)

    card_pred = []
    for i in range(start_idx, start_idx + test_num):
        if i % 1000 == 0:
            print(i)
        cols = [sample_table.columns[sample_table.ColumnIndex(col)] for col in train_data_raw['column'][i]]
        ops = train_data_raw['operator'][i]
        vals = train_data_raw['val'][i]
        est = Query(sample_table, cols, ops, vals)
        if est == 0:
            est = 1
        card_pred.append(est / sample_table.cardinality)

    q_error_list = get_qerror(np.array(card_pred), np.array(train_data_raw['card'][start_idx:start_idx+test_num]))
    print("q error on generated dataset:")
    print("Max q error: {}".format(np.max(q_error_list)))
    print("99 percentile q error: {}".format(np.percentile(q_error_list, 99)))
    print("95 percentile q error: {}".format(np.percentile(q_error_list, 95)))
    print("90 percentile q error: {}".format(np.percentile(q_error_list, 90)))
    print("75 percentile q error: {}".format(np.percentile(q_error_list, 75)))
    print("50 percentile q error: {}".format(np.percentile(q_error_list, 50)))
    print("Average q error: {}".format(np.mean(q_error_list)))

# sample_frame.to_csv('./census_results/train_num_{}.csv'.format(train_num), index=False)

