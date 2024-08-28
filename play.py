import numpy as np
import tensorflow as tf

# Função para imprimir o tabuleiro
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    board_str = [symbols[cell] for cell in board]
    print(f"{board_str[0]} | {board_str[1]} | {board_str[2]}")
    print("--+---+--")
    print(f"{board_str[3]} | {board_str[4]} | {board_str[5]}")
    print("--+---+--")
    print(f"{board_str[6]} | {board_str[7]} | {board_str[8]}")

# Função para verificar se há um vencedor
def check_winner(board):
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colunas
        [0, 4, 8], [2, 4, 6]              # Diagonais
    ]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
            return board[combo[0]]
    if 0 not in board:
        return 0  # Empate
    return None  # Jogo continua

# Função para verificar se o movimento é válido
def is_valid_move(board, move):
    try:
        return board[move] == 0
    except:
        return 0

# Função para a jogada do humano
def human_move(board):
    while True:
        try:
            move = int(input("Escolha uma posição (0-8): "))
            if is_valid_move(board, move):
                return move
            
            print("Movimento inválido. Tente novamente.")
        except:
            print("Movimento inválido. Tente novamente.")

# Função de predição da jogada pela rede neural
def predict_move(model, board):
    board_input = np.array(board).reshape(1, -1)
    prediction = model.predict(board_input)
    move = np.argmax(prediction)
    while not is_valid_move(board, move):
        prediction[0][move] = -1  # Penalize o movimento inválido
        move = np.argmax(prediction)
    return move

# Função para jogar o jogo
def play_game(model):
    board = np.zeros(9, dtype=int)
    print("Você é X e joga primeiro.")
    
    while True:
        print_board(board)
        
        # Jogada do humano
        move = human_move(board)
        board[move] = 1
        
        if check_winner(board) == 1:
            print_board(board)
            print("Você venceu!")
            break
        elif check_winner(board) == 0:
            print_board(board)
            print("Empate!")
            break
        
        # Jogada da rede neural
        move = predict_move(model, board)
        board[move] = -1
        
        if check_winner(board) == -1:
            print_board(board)
            print("A rede neural venceu!")
            break
        elif check_winner(board) == 0:
            print_board(board)
            print("Empate!")
            break

# Carregar o modelo treinado
model = tf.keras.models.load_model('tic_tac_toe_model.h5')

# Jogar o jogo
play_game(model)
