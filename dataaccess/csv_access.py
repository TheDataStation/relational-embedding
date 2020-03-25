import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import itertools
import re


def iterate_rows(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, el in dataframe.iterrows():
        row_values = [el[c] for c in columns]
        yield row_values


def iterate_cells_row_order(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, el in dataframe.iterrows():
        for c in columns:
            cell_value = el[c]
            yield cell_value


## OLD ifaces

def iterate_over_qa(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    ref_column = columns[0]  # basic assumption that first column is the key of interest
    for index, row in dataframe.iterrows():
        q1 = row[ref_column]
        for c in columns[1:]:
            q2 = c
            a = row[c]
            if q2 == "" or a == "":
                continue  # skip empty values
            yield q1, q2, a  # normal qa # TODO: play with this
            # yield q2, a, q1  # inverse qa


def read_csv_file(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    return dataframe


def _iterate_columns_no_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for c in columns:
        clean_tokens = []
        data = dataframe[c]
        for el in data:
            if type(el) is str:
                ct = tp.tokenize(el, " ")
                for t in ct:
                    clean_tokens.append(t)
        tuple = ','.join(clean_tokens)
        yield tuple


def iterate_pairs(path, token_joiner=",", verbose=False):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    df_len = len(dataframe)
    buffer_size = 4
    buffer = None

    for i in range(df_len):
        #buffer = dataframe.iloc[i:i+buffer_size]
        #buffer = dataframe[i:(i + int(df_len/10))]
        buffer = dataframe

        row_buffer = dataframe.iloc[i]

        # generate pairs
        ref_column = columns[0]
        for b in columns:
            ra = str(row_buffer[ref_column])
            rb = str(row_buffer[b])
            if ra == rb:
                continue
            if re.search('[0-9]', ra) is not None or re.search('[0-9]', rb) is not None:
                continue
            # positive sample
            yield ra, rb, 0

            # generate negative samples against rows in buffer
            for idx in range(len(buffer)):
                row = buffer.iloc[idx]
                rc = str(row[ref_column])
                rd = str(row[b])

                if ra == rc or rb == rd:
                    continue
                # negative samples
                # yield ra, rc, 1  # note these are symmetric
                yield ra, rd, 1


def ___iterate_pairs(path, token_joiner=",", verbose=False):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    buffer_size = 4
    buffer = []
    for index, row_buffer in dataframe.iterrows():
        if len(buffer) != buffer_size:  # first iteration, we fill the buffer
            buffer.append(row_buffer)
            continue
        # generate pairs
        ref_column = columns[0]
        for b in columns:
            ra = str(row_buffer[ref_column])
            rb = str(row_buffer[b])
            if ra == rb:
                continue
            if re.search('[0-9]', ra) is not None or re.search('[0-9]', rb) is not None:
                continue
            # positive sample
            yield ra, rb, 0

            # generate negative samples against rows in buffer
            for row in buffer:
                rc = str(row[ref_column])
                rd = str(row[b])

                if ra == rc or rb == rd:
                    continue
                # negative samples
                #yield ra, rc, 1  # note these are symmetric
                yield ra, rd, 1
        buffer = []  # force reinitialization with next cols


def __iterate_pairs(path, token_joiner=",", verbose=False):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    row_buffer = None
    for index, row in dataframe.iterrows():
        if row_buffer is None:  # first iteration, we fill the buffer
            row_buffer = row
            continue
        # generate pairs
        ref_column = columns[0]
        for b in columns:
            ra = str(row_buffer[ref_column])
            rb = str(row_buffer[b])
            if ra == rb:
                continue
            if re.search('[0-9]', ra) is not None or re.search('[0-9]', rb) is not None:
                continue
            # positive sample
            yield ra, rb, 0

            rc = str(row[ref_column])
            rd = str(row[b])

            if ra == rc or rb == rd:
                continue
            # negative samples
            #yield ra, rc, 1  # note these are symmetric
            yield ra, rd, 1
        row_buffer = row


def iterate_columns_no_header(path, token_joiner=",", verbose=False):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for c in columns:
        # clean_tokens = []
        if verbose:
            print("Col: " + str(c))
        data = dataframe[c]
        col = []
        for el in data:
            if type(el) is str:
                el = el.replace(",", ' ')
                ct = tp.tokenize(el, " ")
                tuple = token_joiner.join(ct)
                col.append(tuple)
        yield col


def iterate_columns_with_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for c in columns:
        clean_tokens = []
        data = dataframe[c]
        for el in data:
            if type(el) is str:
                ct = tp.tokenize(el, " ")
                for t in ct:
                    clean_tokens.append(t)
        tuple = ','.join(clean_tokens)
        col_header = []
        clean_tokens = tp.tokenize(c, " ")
        for ct in clean_tokens:
            col_header.append(ct)
        header = ','.join(clean_tokens)
        yield tuple, header


def get_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    clean_tokens = []
    for c in columns:
        ct = tp.tokenize(c, " ")
        for t in ct:
            clean_tokens.append(t)
    clean_tuple = ','.join(clean_tokens)
    return clean_tuple


def iterate_rows_no_header(path, token_joiner=","):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, el in dataframe.iterrows():
        row = []
        for c in columns:
            value = el[c]
            if type(value) is str:
                value = value.replace(",", ' ')
                ct = tp.tokenize(value, " ")
                tuple = token_joiner.join(ct)
                row.append(tuple)
        yield row


def iterate_rows_with_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, el in dataframe.iterrows():
        row = []
        for c in columns:
            value = el[c]
            if type(value) is str:
                ct = tp.tokenize(value, " ")
                for t in ct:
                    row.append(t)
        tuple = ','.join(row)
        yield tuple


def csv_iterator(path, joiner=','):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, row in dataframe.iterrows():
        tuple_list = []
        for col in columns:
            tuple_list.append(str(row[col]))
        tuple = joiner.join(tuple_list)
        yield tuple


def csv_iterator_filter_digits(path, joiner=','):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    for index, row in dataframe.iterrows():
        tuple_list = []
        for col in columns:
            candidate = str(row[col])
            if re.search('[0-9]', candidate) is not None:
                continue
            if len(candidate) < 3:
                continue
            candidate = candidate.lower()
            tuple_list.append(candidate)
        tuple = joiner.join(tuple_list)
        yield tuple


def csv_iterator_with_header(path):
    dataframe = pd.read_csv(path, encoding='latin1')
    columns = dataframe.columns
    clean_col_tokens = set()
    for c in columns:
        toks = tp.tokenize(c, " ")
        for t in toks:
            clean_col_tokens.add(t)
    for index, row in dataframe.iterrows():
        tuple_list = []
        for col in columns:
            tuple_list.append(str(row[col]))
        for c in clean_col_tokens:
            tuple_list.append(c)
        tuple = ','.join(tuple_list)
        yield tuple


def csv_iterator_yield_row_combinations(path, dataframe=None, num_combinations=2, with_header=True):
    if dataframe is None:
        dataframe = pd.read_csv(path, encoding='latin1')

    def combinations_per_row(columns, row, num_combinations=num_combinations):
        tuples = []
        for a, b in itertools.combinations(columns, num_combinations):
            if with_header:
                tuple_tokens = [str(a), str(row[a]), str(b), str(row[b])]
            else:
                tuple_tokens = [str(row[a]), str(row[b])]
            tuple = ' '.join(tuple_tokens)
            tuples.append(tuple)
        return tuples

    for index, row in dataframe.iterrows():
        combinations = combinations_per_row(dataframe.columns, row)
        for c in combinations:
            yield c


def list_files_in_directory(path):
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

if __name__ == "__main__":
    print("CSV Access")

    # for t in csv_iterator_yield_row_combinations("/Users/ra-mit/data/mitdwhdata/Student_department.csv"):
    #     print(t)

    gen = iterate_over_qa("/Users/ra-mit/data/mitdwhdata/col_sample_drupal_employee_directory.csv")
    for q1, q2, a in gen:
        print(str(q1) + " - " + str(q2) + "?: " + str(a))
    exit()

    def compute_num_terms():
        map = defaultdict(int)
        files = list_files_in_directory("/Users/ra-mit/data/mitdwhdata")
        total_files = len(files)
        iteration = 0
        for f in files:
            print("Processing: " + str(f))
            df = read_csv_file(f)
            columns = df.columns
            for c in columns:
                clean_tokens = tp.tokenize(c, " ")
                for ct in clean_tokens:
                    map[ct] += 1
            print(str(iteration) + "/" + str(total_files))
            iteration += 1
            #if iteration > 5:
            #    continue
            print("Size: " + str(len(map)))
            it = csv_iterator(f)
            for tuple in it:
                clean_tokens = tp.tokenize(tuple, ",")
                for ct in clean_tokens:
                    map[ct] += 1
        ordered = sorted(map.items(), key=lambda x: x[1], reverse=True)
        for el in ordered:
            print(str(el))

    files = list_files_in_directory("/Users/ra-mit/data/mitdwhdata")
    for f in files:
        print(str(f))

    example = files[0]

    print(example)

    it = csv_iterator_with_header(example)

    from preprocessing import text_processor as tp

    for tuple in it:
        clean_tokens = tp.tokenize(tuple, ",")
        print(str(clean_tokens))

    # print("Computing number of terms...")
    #
    # compute_num_terms()
    #
    # print("Computing number of terms...OK")
