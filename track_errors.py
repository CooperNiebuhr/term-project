# track_errors.py
import pandas as pd
from load_pgn.config       import PGN_FOLDER, OUTPUT_CSV
from load_pgn.pgn_parser   import parse_pgns
from load_pgn.evaluator    import EngineEvaluator
from load_pgn.categorizer  import score_to_centipawns, categorize_move

def main():
    engine = EngineEvaluator()
    rows = []

    for fen, played_move in parse_pgns(PGN_FOLDER):
        best_move, sb, sp = engine.evaluate(fen, played_move)
        cp_best   = score_to_centipawns(sb)
        cp_played = score_to_centipawns(sp)
        delta     = cp_best - cp_played
        category  = categorize_move(delta, sb, sp)

        # only keep actual “errors”
        if category in ('blunder', 'inaccuracy', 'missed_mate'):
            rows.append({
                'fen': fen,
                'played_move': played_move,
                'best_move': best_move,
                'delta': delta,
                'category': category
            })

    engine.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} errors to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
