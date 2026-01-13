import os
import pandas as pd

def save_results_table(rows: list, out_xlsx: str):
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return df
