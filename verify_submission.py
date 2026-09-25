from pathlib import Path
import sqlite3
import subprocess
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
errors=[]

# Module 1
sql_file=(ROOT/'data_pipeline/sql/required_queries.sql').read_text().upper()
for token in ['SELECT','WHERE','ORDER BY','LIMIT','DISTINCT','BETWEEN',' IN ','JOIN']:
    if token not in sql_file: errors.append(f'M1 missing SQL keyword: {token.strip()}')
with sqlite3.connect(ROOT/'data_pipeline/data/zepto_books.db') as con:
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'books','categories'} <= tables: errors.append('M1 missing normalized tables')
    n=con.execute('SELECT COUNT(*) FROM books').fetchone()[0]
    cats=con.execute('SELECT COUNT(*) FROM categories').fetchone()[0]
    if n<60: errors.append(f'M1 demo database has only {n} rows')
    if cats<3: errors.append(f'M1 demo database has only {cats} categories')

# Module 2
csv=pd.read_csv(ROOT/'analytics/titanic.csv')
expected=['survived','pclass','sex','age','sibsp','parch','fare','embarked','class','who','adult_male','deck','embark_town','alive','alone']
if csv.columns.tolist()!=expected: errors.append('M2 titanic.csv schema mismatch')
correlation=pd.read_csv(ROOT/'analytics/outputs/correlation_matrix.csv',index_col=0)
if correlation.shape!=(6,6) or correlation.columns.tolist()!=expected[:1]+['pclass','age','sibsp','parch','fare']:
    errors.append('M2 correlation matrix is not exactly the required six columns')
pipe=joblib.load(ROOT/'analytics/outputs/best_pipeline.joblib')
sample=csv[['pclass','sex','age','sibsp','parch','fare','embarked']].head(2)
try: pipe.predict(sample)
except Exception as e: errors.append(f'M2 saved pipeline cannot predict on raw input: {e}')

# Module 3
if len(list((ROOT/'support_assistant/docs').glob('doc_*.txt')))!=8: errors.append('M3 corpus is not exactly 8 docs')
prompt=(ROOT/'support_assistant/prompts.py').read_text()
for token in ['ROLE:','CONTEXT:','TASK:','FORMAT:','LENGTH:','NEGATIVE CONSTRAINT:','FEW-SHOT EXAMPLE:']:
    if token not in prompt: errors.append(f'M3 prompt missing {token}')
main=(ROOT/'support_assistant/main.py').read_text()
for token in ['classify_intent','retrieve_and_answer','direct_answer','add_conditional_edges','MOCK_LLM','AnswerResponse']:
    if token not in main: errors.append(f'M3 code missing {token}')

# Git workflow
graph=subprocess.check_output(['git','log','--graph','--all','--oneline'],cwd=ROOT,text=True)
if 'merge: analytics and pipeline feature work' not in graph: errors.append('Git merge commit requirement not found')

if errors:
    print('FAIL')
    print('\n'.join('- '+x for x in errors))
    raise SystemExit(1)
print('PASS — structural checks completed successfully.')
