import pygame
import chess

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
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess GUI")
    board = chess.Board()

    selected_square = None
    running = True

    while running:
        draw_board(screen, board)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                square = chess.square(col, row)

                if selected_square is None:
                    if board.piece_at(square) and board.piece_at(square).color == board.turn:
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

    pygame.quit()

if __name__ == "__main__":
    main()
