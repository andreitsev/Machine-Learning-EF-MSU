from pprint import pprint
from datetime import datetime, date
import typing as t 
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

StudentID = t.Tuple[str, int]
StudentIDs = t.List[StudentID]
"""(fio, group)"""

def get_student_ids_from_google_sheet(
    course_scores_table_url: str='https://docs.google.com/spreadsheets/d/19h8UP5Dku79TxvTiRd03C7g8hE6BoM2bns0gwagKaCQ/edit?usp=sharing',
    fio_col: str='ФИО',
    group_col: str='Группа'
) -> StudentIDs:
    sheet_url = course_scores_table_url.replace("edit?usp=sharing", "gviz/tq?tqx=out:csv")
    df = pd.read_csv(sheet_url)
    df = df.loc[~((df[fio_col].isnull()) | (df[group_col].isnull()))].reset_index()
    df[group_col] = df[group_col].apply(lambda x: int(x[1:]) if str(x).strip().lower().startswith('э') else int(x))
    return [(fio, int(group)) for fio, group in df[[fio_col, group_col]].itertuples(index=False)]


def get_google_form_table_from_url(
    google_form_url: str
) -> pd.DataFrame:
    assert google_form_url.endswith('edit?usp=sharing'), (
        "google_form_url should end with 'edit?usp=sharing'!"
    )
    google_form_url = google_form_url.replace("edit?usp=sharing", "gviz/tq?tqx=out:csv")
    df = pd.read_csv(google_form_url)
    return df

def _process_fio_column(
    fio_series: pd.Series,
) -> pd.Series:
    return fio_series.apply(lambda x: str(x).strip()).astype(str)

def _process_group_column(
    group_series: pd.Series
) -> pd.Series:
    groups_processed = []
    for group in group_series:
        if str(group).lower().strip().startswith('э'):
            group = group[1:]
        if str(group).lower().strip().startswith('-'):
            group = group[1:]
        
        if group is None or np.isnan(group):
            current_group_processed = -1
            
        elif str(group).isdigit():
            current_group_processed = int(group)
        elif isinstance(group, float):
            current_group_processed = int(group) 
        elif str(group)[-3:].strip().isdigit():
            current_group_processed = int(str(group)[-3:].strip())
        else:
            current_group_processed = -1
        groups_processed.append(current_group_processed)
    return pd.Series(groups_processed).astype(int)

def _process_time_column(time_column_series: pd.Series, format: str='date') -> pd.Series:
    assert format in ['date', 'unix'], (
        "format can only be 'unix' or 'date'!"
    )
    return pd.Series([
        datetime.strptime(val, '%d.%m.%Y %H:%M:%S') 
        if format == 'date'
        else datetime.fromtimestamp(int(val))
        for val in time_column_series
    ])
    
def _process_score_column(
    score_series: pd.Series
) -> pd.Series:
    return pd.Series([float(str(val).strip().split('/')[0]) for val in score_series])
    

def process_google_form_table(
    google_form_table: pd.DataFrame,
    fio_col: str,
    group_col: str,
    end_time_col: str,
    score_col: str,
    start_time_col: t.Optional[str]=None,
) -> pd.DataFrame:
    
    processed_google_form_table = google_form_table.copy(deep=True)
    ### processing fio col
    processed_google_form_table[fio_col] = _process_fio_column(processed_google_form_table[fio_col])
    ### processing group col
    processed_google_form_table[group_col] = _process_group_column(processed_google_form_table[group_col])
    ### processing end time col
    processed_google_form_table[end_time_col] = _process_time_column(
        time_column_series=processed_google_form_table[end_time_col],
        format='date'
    )
    ### processing start time col
    if start_time_col is not None:
        processed_google_form_table[start_time_col] = _process_time_column(
            time_column_series=processed_google_form_table[end_time_col],
            format='unix'
        )
    ### processing score col    
    processed_google_form_table[score_col] = _process_score_column(score_series=processed_google_form_table[score_col])
    return processed_google_form_table


def linear_score_punisher(
    end_time_series: pd.Series,
    deadline_datetime: datetime,
    punish_coefficient: float=1.0,
) -> t.Tuple[pd.Series]:
    """
    Returns pd.Series with amount of scores to substract from main score column
    because of deadline crossing and series with minutes after deadline
    """
    minutes_after_deadline = (end_time_series - deadline_datetime).dt.total_seconds() // 60
    total_score_to_extract = punish_coefficient*minutes_after_deadline
    return total_score_to_extract.clip(lower=0), minutes_after_deadline.clip(lower=0)
    


def get_theoretical_tests_scores(
    test_date: date,
    main_table_fio_group_list: StudentIDs,
    google_form_url: t.Optional[str]=None,
    google_form_table: t.Optional[pd.DataFrame]=None,
    google_form_student_fio_col: str='Фамилия<пробел>Имя',
    google_form_student_group_col: str='Группа',
    google_form_start_time_col: t.Optional[str]=None,
    google_form_end_time_col: str='Отметка времени',
    google_form_score_col: str='Баллы',
    deadline_datetime: t.Optional[datetime]=None,
    plot_intersection: bool=False,
) -> pd.DataFrame:
    
    assert ((int(google_form_url is not None) + int(google_form_table is not None)) % 2 != 0), (
        "At least one (and maximum one!) of google_form_url or google_form_table should be provided!"
    )
    
    if google_form_table is None:
        google_form_table = get_google_form_table_from_url(google_form_url=google_form_url)
    
    #### processing google_form_table
    google_form_table = process_google_form_table(
        google_form_table=google_form_table,
        fio_col=google_form_student_fio_col,
        group_col=google_form_student_group_col,
        start_time_col=google_form_start_time_col,
        end_time_col=google_form_end_time_col,
        score_col=google_form_score_col,
    )
    google_form_table = google_form_table.loc[(google_form_table[google_form_end_time_col].dt.date == test_date)].reset_index(drop=True)
    google_form_table_columns = [
            google_form_student_fio_col,
            google_form_student_group_col,
            google_form_end_time_col,
            google_form_score_col,
        ]
    if google_form_start_time_col is not None:
        google_form_table_columns.append(google_form_start_time_col)

    #### substract some score because of deadline
    if deadline_datetime is not None:
        scores_to_extract, minutes_after_deadline = linear_score_punisher(
            end_time_series=google_form_table[google_form_end_time_col],
            deadline_datetime=deadline_datetime,
            punish_coefficient=1.0,
        )
        google_form_table['minutes_after_deadline'] = minutes_after_deadline
        google_form_table['score_with_punishment'] = (google_form_table[google_form_score_col] - scores_to_extract).clip(lower=0)
        google_form_table_columns.extend(['minutes_after_deadline', 'score_with_punishment'])
    
    
    all_students_df = pd.DataFrame(main_table_fio_group_list, columns=['fio', 'group']).astype({'fio': 'string', "group": 'int'})
    all_students_df = (
        all_students_df
        .merge(
            google_form_table[google_form_table_columns]
            .rename(columns={
                google_form_student_fio_col: 'fio', 
                google_form_student_group_col: 'group',
                google_form_score_col: 'score',
                google_form_end_time_col: 'end_time'
            })
            ,
            on=['fio', 'group'],
            how='left'
        )
        .sort_values(by=['group', 'fio'], ascending=True)
    )
    
    if plot_intersection:
        students_in_test = [(fio, group) for fio, group in google_form_table[[google_form_student_fio_col, google_form_student_group_col]].itertuples(index=False)]
        print(f"#students in test: {len(students_in_test):,}")
        print(f"#students in main course table: {len(main_table_fio_group_list):,}")
        print(f"test - main course table: {len(set(students_in_test) - set(main_table_fio_group_list)):,}")
        pprint(set(students_in_test) - set(main_table_fio_group_list))
        plt.figure(figsize=(6, 6))
        venn2(
            subsets=[
                set(students_in_test),
                set(main_table_fio_group_list)
            ],
            set_labels=['test', 'all']
        );
        plt.show()
        
    return all_students_df