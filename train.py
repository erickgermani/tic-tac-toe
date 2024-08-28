import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Função para verificar se o estado do tabuleiro é válido
def is_valid_board(board):
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    return (x_count == o_count) or (x_count == o_count + 1)

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
    return None

# Função para avaliar um movimento
def evaluate_move(board, move, player):
    board[move] = player
    winner = check_winner(board)
    board[move] = 0  # Desfazer o movimento
    if winner == player:
        return 10  # Vitória
    elif winner == -player:
        return -10  # Derrota
    else:
        # Avaliação com base em possibilidades futuras
        opponent = -player
        opponent_moves = [m for m in range(9) if board[m] == 0]
        score = 0
        for opponent_move in opponent_moves:
            board[opponent_move] = opponent
            if check_winner(board) == opponent:
                score -= 5  # Penalidade por deixar o oponente vencer
            board[opponent_move] = 0  # Desfazer o movimento
        return score

# Função para determinar o melhor movimento baseado na avaliação
def best_move(board, player):
    best_score = -np.inf
    best_move = None
    for move in range(9):
        if board[move] == 0:
            score = evaluate_move(board, move, player)
            if score > best_score:
                best_score = score
                best_move = move
    return best_move

# Função para gerar dados de treinamento estratégicos
def generate_strategic_training_data():
    X = []
    y = []
    for i in range(19683):  # 3^9 possíveis estados (vazio, X, O)
        board = np.base_repr(i, base=3).zfill(9)
        board = np.array([int(x) - 1 for x in board])
        if is_valid_board(board):
            move = best_move(board, 1)
            if move is not None:
                X.append(board)
                y.append(move)
    return np.array(X), np.array(y)

# Gerar os dados de treinamento
X_train, y_train = generate_strategic_training_data()

# Converter y_train para categórico
y_train = tf.keras.utils.to_categorical(y_train, num_classes=9)

# Criar o modelo
model = Sequential([
    Dense(128, input_dim=9, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Salvar o modelo treinado
model.save('tic_tac_toe_model.h5')

print("Modelo treinado e salvo como 'tic_tac_toe_model.h5'")
