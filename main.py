import pygame
import chess
import random
from ai import select_random_move

# Constants
WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# Load pieces
PIECES = {}
for color in ['w', 'b']:
    for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
        filename = f"{color}{piece}.png"
        PIECES[color + piece] = pygame.transform.scale(
            pygame.image.load(f"pieces/{filename}"), (SQUARE_SIZE, SQUARE_SIZE)
        )

def draw_board(screen, board):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(screen, color, pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            symbol = piece.symbol()
            key = ('w' if symbol.isupper() else 'b') + symbol.upper()
            screen.blit(PIECES[key], pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def main():
    # Ask player to choose a side
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

    board = chess.Board()

    # If human is black, let bot make the first move
    if human_color == chess.BLACK:
        bot_move = select_random_move(board)
        if bot_move:
            board.push(bot_move)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess GUI")

    selected_square = None
    running = True

    while running:
        draw_board(screen, board)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Only accept human moves on their turn
                if board.turn != human_color:
                    continue

                x, y = pygame.mouse.get_pos()
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                square = chess.square(col, row)

                if selected_square is None:
                    # Select a piece
                    if board.piece_at(square) and board.piece_at(square).color == human_color:
                        selected_square = square
                else:
                    # Handle pawn promotion
                    promotion = None
                    piece = board.piece_at(selected_square)
                    dest_rank = square // 8
                    if piece and piece.piece_type == chess.PAWN and dest_rank in (0, 7):
                        choice = input("Promote to (q, r, b, n): ").strip().lower()
                        promo_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
                        promotion = promo_map.get(choice, chess.QUEEN)
                    # Create move with or without promotion
                    if promotion:
                        move = chess.Move(selected_square, square, promotion=promotion)
                    else:
                        move = chess.Move(selected_square, square)
                    # Execute move if legal
                    if move in board.legal_moves:
                        board.push(move)
                        # Bot responds
                        if not board.is_game_over():
                            bot_move = select_random_move(board)
                            if bot_move:
                                board.push(bot_move)
                    selected_square = None

    pygame.quit()

if __name__ == "__main__":
    main()
