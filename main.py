import pygame
import chess
import random
from ai import select_greedy_move
from utils import get_game_status

# Constants
WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# Load piece images (named wP.png, bP.png, etc.)
PIECES = {}
for color in ['w', 'b']:
    for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
        filename = f"{color}{piece}.png"
        img = pygame.image.load(f"pieces/{filename}")
        PIECES[color + piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board):
    # Draw squares
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE,
                               SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            sym = piece.symbol()
            key = ('w' if sym.isupper() else 'b') + sym.upper()
            screen.blit(PIECES[key],
                        pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE,
                                    SQUARE_SIZE, SQUARE_SIZE))

def main():
    # 1. Choose side
    choice = input("Choose side (white/black/random): ").strip().lower()
    if choice == 'white':
        human_color = chess.WHITE
    elif choice == 'black':
        human_color = chess.BLACK
    elif choice == 'random':
        human_color = random.choice([chess.WHITE, chess.BLACK])
        print(f"Randomly assigning you {'white' if human_color == chess.WHITE else 'black'}.")
    else:
        print("Invalid choice, defaulting to white.")
        human_color = chess.WHITE

    # 2. Initialize board and possibly let bot open
    board = chess.Board()
    if human_color == chess.BLACK:
        bot_move = select_greedy_move(board)
        if bot_move:
            board.push(bot_move)

    # 3. Initialize Pygame
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess GUI")
    font = pygame.font.SysFont(None, 36)

    selected_square = None
    running = True
    game_over = False
    result_text = ""

    # 4. Main loop
    while running:
        draw_board(screen, board)

        if game_over:
            # Display end-of-game message
            text = font.render(result_text, True, (255, 0, 0))
            rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(text, rect)
            pygame.display.flip()

            # Only handle quit events so user can read the result
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            continue

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Only allow human to move on their turn
                if board.turn != human_color:
                    continue

                x, y = pygame.mouse.get_pos()
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                square = chess.square(col, row)

                # Select or move piece
                if selected_square is None:
                    if (board.piece_at(square) and
                        board.piece_at(square).color == human_color):
                        selected_square = square
                else:
                    # Handle pawn promotion
                    promotion = None
                    piece = board.piece_at(selected_square)
                    dest_rank = square // 8
                    if (piece and piece.piece_type == chess.PAWN and
                        dest_rank in (0, 7)):
                        choice_p = input("Promote to (q, r, b, n): ").strip().lower()
                        promo_map = {'q': chess.QUEEN,
                                     'r': chess.ROOK,
                                     'b': chess.BISHOP,
                                     'n': chess.KNIGHT}
                        promotion = promo_map.get(choice_p, chess.QUEEN)

                    # Build and push the move
                    move = (chess.Move(selected_square, square, promotion=promotion)
                            if promotion else
                            chess.Move(selected_square, square))
                    if move in board.legal_moves:
                        board.push(move)

                        # Check for game end
                        status = get_game_status(board)
                        if status:
                            game_over = True
                            result_text = status
                        else:
                            # Bot'pgns turn
                            bot_move = select_greedy_move(board)
                            if bot_move:
                                board.push(bot_move)
                                status = get_game_status(board)
                                if status:
                                    game_over = True
                                    result_text = status

                    selected_square = None

    pygame.quit()

if __name__ == "__main__":
    main()
