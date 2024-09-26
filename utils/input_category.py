import pandas as pd


def category_trans_by_logic(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    set1 = {'workday', 'holiday', 'day_of_week'}
    set2 = set(data.columns)

    assert set1.issubset(set2), "['workday', 'holiday', 'day_of_week'] should be in data.columns"

    category_list = []
    for i in range(len(data)):
        if data['workday'][i] == 1:
            category_list.append(0)
        elif data['holiday'][i] == 1 or (
                data['day_of_week'][i] == 7 and data['workday'][i] == 0):
            category_list.append(1)
        else:
            category_list.append(2)
    data['input_category'] = category_list
    return data