import pandas as pd
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import CroppedTable, AutoTableDetector
from gmft_pymupdf import PyMuPDFDocument
from gmft.auto import AutoFormatConfig, AutoTableFormatter

detector = AutoTableDetector()

def ingest_pdf(pdf_path, start_page, end_page) -> list[CroppedTable]:
    doc = PyMuPDFDocument(pdf_path)

    tables = []
    for i in range(start_page, end_page):
        tables += detector.extract(doc.get_page(i))
    return tables, doc
        
def merge_rows(df):
    """
    Processes the DataFrame row by row. For each row (starting from the second row),
    if either the label (col 0) or notes (col 6) is not None and the material (col 1) is None:
        - If the current row's label is not None, it is appended to the previous row's label.
        - If the current row's notes is not None, it is appended to the previous row's notes.
        - The current row is then removed.
    
    Columns are defined by index:
        0: label
        1: material
        6: notes
        
    Returns:
        A new DataFrame with the rows merged and the index reset.
    """
    drop_indices = []
    
    new_df = df.copy(deep=True)
    
    # Iterate from the second row to the end
    for i in range(1, len(new_df)):
        # Check if either label (col 0) or notes (col 6) is not missing and material (col 1) is missing.
        if ((pd.notnull(new_df.iat[i, 0]) or pd.notnull(new_df.iat[i, 6]))
                and pd.isnull(new_df.iat[i, 1])):
            
            # Append current row's label (col 0) to previous row's label if not missing.
            if pd.notnull(new_df.iat[i, 0]):
                if pd.isnull(new_df.iat[i - 1, 0]):
                    new_df.iat[i - 1, 0] = new_df.iat[i, 0]
                else:
                    new_df.iat[i - 1, 0] = str(new_df.iat[i - 1, 0]) + " " + str(new_df.iat[i, 0])
            
            # Append current row's notes (col 6) to previous row's notes if not missing.
            if pd.notnull(new_df.iat[i, 6]):
                if pd.isnull(new_df.iat[i - 1, 6]):
                    new_df.iat[i - 1, 6] = new_df.iat[i, 6]
                else:
                    new_df.iat[i - 1, 6] = str(new_df.iat[i - 1, 6]) + " " + str(new_df.iat[i, 6])
            
            # Mark the current row for removal.
            drop_indices.append(i)
    
    # Remove the marked rows and reset the index.
    new_df = new_df.drop(drop_indices).reset_index(drop=True)
    return new_df

config = AutoFormatConfig(verbosity=3, semantic_spanning_cells=True)
formatter = AutoTableFormatter(config=config)
tables, doc = ingest_pdf('HMOF_P_12X40_FullPlans.pdf', 10,15)

for table in tables:
    ft = formatter.format(table)
    df = ft.df()
    df2 = merge_rows(df)
    df2.to_csv('page_' + str(ft.page.page_number+1) + '.csv', index=False)

