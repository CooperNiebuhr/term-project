import ast
import pandas as pd

THEME_MAPPING = {
    'mateIn2': 'mateIn2',
    'backRankMate': 'backRankMate',
    'fork': 'fork',
    'pin': 'pin',
    'skewer': 'skewer',
    'discoveredAttack': 'discoveredAttack',
    'deflection': 'deflection',
    'hangingPiece': 'hangingPiece',
    'promotion': 'promotion',
    'sacrifice': 'sacrifice'
}


INPUT_CSV = 'data/lichess_db_puzzle.csv'



df = pd.read_csv(INPUT_CSV, usecols=['FEN', 'Moves', 'Rating', 'Themes'])


def simplify_themes(theme_str: str) -> str:
    if not isinstance(theme_str, str): 
        return 'other'
    for tag in theme_str.split():
        if tag in THEME_MAPPING:
            return THEME_MAPPING[tag]
    return 'other'

df['simple_label'] = df['Themes'].apply(simplify_themes)
counts = df['simple_label'].value_counts()
print(counts)
# Example: 5k per class, with replacement for smaller buckets
samples = []
for cls in counts.index:
    group = df[df['simple_label'] == cls]
    samples.append(
        group.sample(n=10000, replace=len(group)<10000, random_state=42)
    )
df_sample = pd.concat(samples).sample(frac=1, random_state=42)




df_sample = pd.concat(samples).sample(frac=1, random_state=42)


df_sample.to_csv("handled_data.csv", index=False)
print("Saved balanced dataset with", len(df_sample), "rows to handled_data.csv")
